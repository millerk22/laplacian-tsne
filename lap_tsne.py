# Authors: The scikit-learn developers + Kevin Miller
# This is code adapted by Kevin Miller (millerk22) to approximately 
# solve the tSNE computation with a graph Laplacian approximation. 

# **Requires the graphlearning package**

import warnings
from numbers import Integral, Real
from time import time

import numpy as np
from scipy import linalg
from scipy.sparse import csr_matrix, issparse, coo_matrix
from scipy.spatial.distance import pdist, squareform
import graphlearning as gl
from joblib import Parallel, delayed
from functools import partial

from sklearn.base import (
    BaseEstimator,
    ClassNamePrefixFeaturesOutMixin,
    TransformerMixin,
    _fit_context,
)
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import _VALID_METRICS, pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
from sklearn.utils._param_validation import Hidden, Interval, StrOptions, validate_params
from sklearn.utils.validation import _num_samples, check_non_negative

import matplotlib.pyplot as plt

# mypy error: Module 'sklearn.manifold' has no attribute '_utils'
# mypy error: Module 'sklearn.manifold' has no attribute '_barnes_hut_tsne'
from sklearn.manifold import _barnes_hut_tsne, _utils  # type: ignore

MACHINE_EPSILON = np.finfo(np.double).eps


def opt_bisection(f, xmin=0.0, xmax=10.0, tol=1e-4, itermax=1000, debug=False):
    r"""
    Function to find root of $f(x)$ via bisection method. Default settings of left and right endpoints
    are with the entropy-perplexity smoothing in mind.
    """

    other_iters = 0
    assert f(xmin) > 0.0
    while f(xmax) > 0.0:
        if f(xmin)*f(xmax) < 0:
            raise ValueError(f"f(xmin) = {f(xmin)} (should be > 0) and f(xmax) ={f(xmax)} (want to be < 0)")
        xmax *= 10
        other_iters += 1
        if other_iters > 10:
            break
    
    if other_iters != 0 and debug:
        print(other_iters)
        
    it = 0
    if abs(f(xmin)) <= tol:
        if debug: print(it)
        return xmin 
    if abs(f(xmax)) <= tol:
        if debug: print(it)
        return xmax
    
    
    while it <= itermax:
        c = 0.5*(xmin + xmax)
        if abs(f(c)) <= tol:
            if debug: print(it)
            return c 
        if f(xmin)*f(c) > 0.0:
            xmin = c 
        else:
            xmax = c 
        it += 1
    print(f"------- WARNING: Bisection did not converge in {itermax} iterations to tol = {tol}, current err = {abs(f(c))} -------")
    return None


def _rpcholesky(X_, k, tol=1e-5, returnG=False):
    n = X_.shape[0]
    rng = np.random.default_rng()
    
    diags = np.ones(n).astype(float)
    orig_trace = float(n) # for the kernel k(x_i, x_j) = 1/(1 + ||x_i - x_j||^2)

    landmarks = []
    G = np.zeros((k,n))

    for i in range(k):
        idx = rng.choice(range(n), p=diags/diags.sum())
        landmarks.append(idx)  # add this pivot to the landmark set
        idx_row = 1. / (1. + pairwise_distances(X_, X_[idx,:].reshape(1, -1), metric="euclidean", squared=True)).flatten()
        G[i,:] = (idx_row - G[:i, idx].T @ G[:i,:]) / np.sqrt(diags[idx])
        diags -= G[i,:]**2.
        diags = diags.clip(min=0)

        if tol > 0.0 and diags.sum() <= tol*orig_trace:
            G = G[:i,:]
            break

    if returnG:
        return landmarks, G
    
    return landmarks




def _laplacian_kl_divergence_eigen(
    params,
    evals,
    n_samples,
    n_components,
    landmarks,
    skip_num_points=0,
    compute_error=True,
    V = None,
    kernel="standard"
):
    r"""t-SNE objective function: gradient of the Laplacian term + a low-rank approximation of the repulsion term

    Parameters
    ----------
    params : ndarray of shape (n_params,)
        Unraveled embedding. This should be of shape (n_components*k_eigen,)

    evals : numpy array of shape (k_eigen,)
        Array of low frequency eigenvalues    
    
    n_samples : int
        Number of samples.

    n_components : int
        Dimension of the embedded space.

    landmarks : list of ints
        Set of indices corresponding to the landmark points to be used for the low-rank approximation of 
        K and K2

    skip_num_points : int, default=0
        This does not compute the gradient for points with indices below
        `skip_num_points`. This is useful when computing transforms of new
        data where you'd like to keep the old data fixed.

    compute_error: bool, default=True
        If False, the kl_divergence is not computed and returns NaN.

    V: ndarray of shape (k_eigen, n_components)
        Eigenvector matrix from the graph Laplacian, columns are the eigenvectors corresponding to the eigenvalues in ``evals``

    Returns
    -------
    lap_kl_divergence : float
        Laplacian approximation of the Kullback-Leibler divergence of p_ij and q_ij.

    grad : ndarray of shape (n_params,)
        Unraveled gradient of the this Laplacian approximation of Kullback-Leibler divergence 
        with respect to the embedding.
    """
    if landmarks is None:
        raise ValueError(f"landmarks variable cannot be None....")
    
    if V is None:
        raise ValueError("Eigenvector matrix ``V`` cannot be None...")

    k_eigen = len(evals)
    A_embedded = params.reshape(k_eigen, n_components)  # these are the eigenfunction coefficients, our optimization variables.
    grad = evals.reshape(1,-1) @ A_embedded / n_samples   # attraction term gradient  
    cost = (A_embedded.T @ grad).sum()        # attraction term cost 
        
    X_embedded_lm = V[landmarks,:] @ A_embedded  # this is the subset of rows of the "full embedding" corresponding to the current 
                                            # eigenfunction coeffs and the landmark points

    # compute the repulsive part of the cost function
    G = 1./ (1. + pairwise_distances(X_embedded_lm, X_embedded_lm, metric="euclidean", squared=True)) 
    repulsive_sum = G.sum()
    cost += np.log(repulsive_sum / (n_samples**2.0))
    
    # compute the repulsive terms for the gradient 
    H = G**2.
    grad += (4.0/repulsive_sum) * (S2 @ X_embedded - S2.sum(axis=1).reshape(-1,1) * X_embedded) 
    Hll_inv = np.linalg.inv(H[landmarks,:])
    H_ = H @ Hll_inv    # repulsive terms squared is H_ @ H.T
    grad += (4.0/repulsive_sum) * (H_ @ (H.T @ X_embedded) - (H_ @ (H.sum(axis=0))).reshape(-1, 1) * X_embedded)
    grad = grad.ravel()
    
    return cost, grad



################################################
################################################
#### CHANGE TO DO ONLY "full" or "landmarks" sampling.
#### SEE WHAT CAN USE FROM Adam's code
#############################################
###########################################

def _gradient_descent(
    objective,
    p0,
    it,
    max_iter,
    n_iter_check=1,
    n_iter_without_progress=300,
    momentum=0.8,
    learning_rate=200.0,
    min_gain=0.01,
    min_grad_norm=1e-7,
    verbose=0,
    num_landmarks=None, 
    rpchol_iter=None,
    args=None,
    kwargs=None,
):
    """Batch gradient descent with momentum and individual gains.

    Parameters
    ----------
    objective : callable
        Should return a tuple of cost and gradient for a given parameter
        vector. When expensive to compute, the cost can optionally
        be None and can be computed every n_iter_check steps using
        the objective_error function.

    p0 : array-like of shape (n_params,)
        Initial parameter vector.

    it : int
        Current number of iterations (this function will be called more than
        once during the optimization).

    max_iter : int
        Maximum number of gradient descent iterations.

    n_iter_check : int, default=1
        Number of iterations before evaluating the global error. If the error
        is sufficiently low, we abort the optimization.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization.

    momentum : float within (0.0, 1.0), default=0.8
        The momentum generates a weight for previous gradients that decays
        exponentially.

    learning_rate : float, default=200.0
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers.

    min_gain : float, default=0.01
        Minimum individual gain for each parameter.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be aborted.

    verbose : int, default=0
        Verbosity level.

    args : sequence, default=None
        Arguments to pass to objective function.

    kwargs : dict, default=None
        Keyword arguments to pass to objective function.

    Returns
    -------
    p : ndarray of shape (n_params,)
        Optimum parameters.

    error : float
        Optimum.

    i : int
        Last iteration.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}
    
    eigen_flag = len(args[0].shape) == 1
    lowrank_flag = (num_landmarks is not None) and (not eigen_flag) # will only do updated RPCholesky for landmark points in the case that we're not doing method = "eigen"

    if lowrank_flag:
        assert rpchol_iter is not None 
        n_samples = args[1]
        n_components = args[2]

    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(float).max
    best_error = np.finfo(float).max
    best_iter = i = it

    iterates = []

    if lowrank_flag:
        # compute initial RPChol landmarks
        landmarks = _rpcholesky(p.reshape(n_samples, n_components), num_landmarks) 
        args.append(landmarks)

    tic = time()
    for i in range(it, max_iter):
        check_convergence = (i + 1) % n_iter_check == 0
        # only compute the error when needed
        kwargs["compute_error"] = check_convergence or i == max_iter - 1

        if lowrank_flag:
            if (i+1) % rpchol_iter == 0:
                # recompute the landmarks
                landmarks = _rpcholesky(p.reshape(n_samples, n_components), num_landmarks)
                args[-1] = landmarks 

        iterates.append(p.copy())
        error, grad = objective(p, *args, **kwargs)

        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update

        if check_convergence:
            toc = time()
            duration = toc - tic
            tic = toc
            grad_norm = linalg.norm(grad)

            if verbose >= 2:
                print(
                    "[t-SNE] Iteration %d: error = %.7f,"
                    " gradient norm = %.7f"
                    " (%s iterations in %0.3fs)"
                    % (i + 1, error, grad_norm, n_iter_check, duration)
                )

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: did not make any progress "
                        "during the last %d episodes. Finished."
                        % (i + 1, n_iter_without_progress)
                    )
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: gradient norm %f. Finished."
                        % (i + 1, grad_norm)
                    )
                break
        
    if verbose >= 2:
        return p, error, i, iterates
    else:
        return p, error, i

@validate_params(
    {
        "X": ["array-like", "sparse matrix"],
        "X_embedded": ["array-like", "sparse matrix"],
        "n_neighbors": [Interval(Integral, 1, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
    },
    prefer_skip_nested_validation=True,
)
def trustworthiness(X, X_embedded, *, n_neighbors=5, metric="euclidean"):
    r"""Indicate to what extent the local structure is retained.

    The trustworthiness is within [0, 1]. It is defined as

    .. math::

        T(k) = 1 - \frac{2}{nk (2n - 3k - 1)} \sum^n_{i=1}
            \sum_{j \in \mathcal{N}_{i}^{k}} \max(0, (r(i, j) - k))

    where for each sample i, :math:`\mathcal{N}_{i}^{k}` are its k nearest
    neighbors in the output space, and every sample j is its :math:`r(i, j)`-th
    nearest neighbor in the input space. In other words, any unexpected nearest
    neighbors in the output space are penalised in proportion to their rank in
    the input space.

    Parameters
    ----------
    X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
        (n_samples, n_samples)
        If the metric is 'precomputed' X must be a square distance
        matrix. Otherwise it contains a sample per row.

    X_embedded : {array-like, sparse matrix} of shape (n_samples, n_components)
        Embedding of the training data in low-dimensional space.

    n_neighbors : int, default=5
        The number of neighbors that will be considered. Should be fewer than
        `n_samples / 2` to ensure the trustworthiness to lies within [0, 1], as
        mentioned in [1]_. An error will be raised otherwise.

    metric : str or callable, default='euclidean'
        Which metric to use for computing pairwise distances between samples
        from the original input space. If metric is 'precomputed', X must be a
        matrix of pairwise distances or squared distances. Otherwise, for a list
        of available metrics, see the documentation of argument metric in
        `sklearn.pairwise.pairwise_distances` and metrics listed in
        `sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS`. Note that the
        "cosine" metric uses :func:`~sklearn.metrics.pairwise.cosine_distances`.

        .. versionadded:: 0.20

    Returns
    -------
    trustworthiness : float
        Trustworthiness of the low-dimensional embedding.

    References
    ----------
    .. [1] Jarkko Venna and Samuel Kaski. 2001. Neighborhood
           Preservation in Nonlinear Projection Methods: An Experimental Study.
           In Proceedings of the International Conference on Artificial Neural Networks
           (ICANN '01). Springer-Verlag, Berlin, Heidelberg, 485-491.

    .. [2] Laurens van der Maaten. Learning a Parametric Embedding by Preserving
           Local Structure. Proceedings of the Twelfth International Conference on
           Artificial Intelligence and Statistics, PMLR 5:384-391, 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_blobs
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.manifold import trustworthiness
    >>> X, _ = make_blobs(n_samples=100, n_features=10, centers=3, random_state=42)
    >>> X_embedded = PCA(n_components=2).fit_transform(X)
    >>> print(f"{trustworthiness(X, X_embedded, n_neighbors=5):.2f}")
    0.92
    """
    n_samples = _num_samples(X)
    if n_neighbors >= n_samples / 2:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) should be less than n_samples / 2"
            f" ({n_samples / 2})"
        )
    dist_X = pairwise_distances(X, metric=metric)
    if metric == "precomputed":
        dist_X = dist_X.copy()
    # we set the diagonal to np.inf to exclude the points themselves from
    # their own neighborhood
    np.fill_diagonal(dist_X, np.inf)
    ind_X = np.argsort(dist_X, axis=1)
    # `ind_X[i]` is the index of sorted distances between i and other samples
    ind_X_embedded = (
        NearestNeighbors(n_neighbors=n_neighbors)
        .fit(X_embedded)
        .kneighbors(return_distance=False)
    )

    # We build an inverted index of neighbors in the input space: For sample i,
    # we define `inverted_index[i]` as the inverted index of sorted distances:
    # inverted_index[i][ind_X[i]] = np.arange(1, n_sample + 1)
    inverted_index = np.zeros((n_samples, n_samples), dtype=int)
    ordered_indices = np.arange(n_samples + 1)
    inverted_index[ordered_indices[:-1, np.newaxis], ind_X] = ordered_indices[1:]
    ranks = (
        inverted_index[ordered_indices[:-1, np.newaxis], ind_X_embedded] - n_neighbors
    )
    t = np.sum(ranks[ranks > 0])
    t = 1.0 - t * (
        2.0 / (n_samples * n_neighbors * (2.0 * n_samples - 3.0 * n_neighbors - 1.0))
    )
    return t


class LaplacianTSNE(ClassNamePrefixFeaturesOutMixin, TransformerMixin, BaseEstimator):
    """T-distributed Stochastic Neighbor Embedding.

    t-SNE [1] is a tool to visualize high-dimensional data. It converts
    similarities between data points to joint probabilities and tries
    to minimize the Kullback-Leibler divergence between the joint
    probabilities of the low-dimensional embedding and the
    high-dimensional data. t-SNE has a cost function that is not convex,
    i.e. with different initializations we can get different results.

    It is highly recommended to use another dimensionality reduction
    method (e.g. PCA for dense data or TruncatedSVD for sparse data)
    to reduce the number of dimensions to a reasonable amount (e.g. 50)
    if the number of features is very high. This will suppress some
    noise and speed up the computation of pairwise distances between
    samples. For more tips see Laurens van der Maaten's FAQ [2].

    Read more in the :ref:`User Guide <t_sne>`.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    perplexity : float, default=30.0
        The perplexity is related to the number of nearest neighbors that
        is used in other manifold learning algorithms. Larger datasets
        usually require a larger perplexity. Consider selecting a value
        between 5 and 50. Different values can result in significantly
        different results. The perplexity must be less than the number
        of samples.

    learning_rate : float or "auto", default="auto"
        The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
        the learning rate is too high, the data may look like a 'ball' with any
        point approximately equidistant from its nearest neighbours. If the
        learning rate is too low, most points may look compressed in a dense
        cloud with few outliers. If the cost function gets stuck in a bad local
        minimum increasing the learning rate may help.
        Note that many other t-SNE implementations (bhtsne, FIt-SNE, openTSNE,
        etc.) use a definition of learning_rate that is 4 times smaller than
        ours. So our learning_rate=200 corresponds to learning_rate=800 in
        those other implementations. The 'auto' option sets the learning_rate
        to `max(N / 4, 50)` where N is the sample size,
        following [4] and [5].

        .. versionchanged:: 1.2
           The default value changed to `"auto"`.

    max_iter : int, default=1000
        Maximum number of iterations for the optimization. Should be at
        least 250.

        .. versionchanged:: 1.5
            Parameter name changed from `n_iter` to `max_iter`.

    n_iter_without_progress : int, default=300
        Maximum number of iterations without progress before we abort the
        optimization, used after 250 initial iterations with early
        exaggeration. Note that progress is only checked every 50 iterations so
        this value is rounded to the next multiple of 50.

        .. versionadded:: 0.17
           parameter *n_iter_without_progress* to control stopping criteria.

    min_grad_norm : float, default=1e-7
        If the gradient norm is below this threshold, the optimization will
        be stopped.

    metric : str or callable, default='euclidean'
        The metric to use when calculating distance between instances in a
        feature array. If metric is a string, it must be one of the options
        allowed by scipy.spatial.distance.pdist for its metric parameter, or
        a metric listed in pairwise.PAIRWISE_DISTANCE_FUNCTIONS.
        If metric is "precomputed", X is assumed to be a distance matrix.
        Alternatively, if metric is a callable function, it is called on each
        pair of instances (rows) and the resulting value recorded. The callable
        should take two arrays from X as input and return a value indicating
        the distance between them. The default is "euclidean" which is
        interpreted as squared euclidean distance.

    metric_params : dict, default=None
        Additional keyword arguments for the metric function.

        .. versionadded:: 1.1

    init : {"random", "pca"} or ndarray of shape (n_samples, n_components), \
            default="pca"
        Initialization of embedding.
        PCA initialization cannot be used with precomputed distances and is
        usually more globally stable than random initialization.

        .. versionchanged:: 1.2
           The default value changed to `"pca"`.

    verbose : int, default=0
        Verbosity level.

    random_state : int, RandomState instance or None, default=None
        Determines the random number generator. Pass an int for reproducible
        results across multiple function calls. Note that different
        initializations might result in different local minima of the cost
        function. See :term:`Glossary <random_state>`.

    method : {'barnes_hut', 'exact'}, default='barnes_hut'
        By default the gradient calculation algorithm uses Barnes-Hut
        approximation running in O(NlogN) time. method='exact'
        will run on the slower, but exact, algorithm in O(N^2) time. The
        exact algorithm should be used when nearest-neighbor errors need
        to be better than 3%. However, the exact method cannot scale to
        millions of examples.

        .. versionadded:: 0.17
           Approximate optimization *method* via the Barnes-Hut.

    angle : float, default=0.5
        Only used if method='barnes_hut'
        This is the trade-off between speed and accuracy for Barnes-Hut T-SNE.
        'angle' is the angular size (referred to as theta in [3]) of a distant
        node as measured from a point. If this size is below 'angle' then it is
        used as a summary node of all points contained within it.
        This method is not very sensitive to changes in this parameter
        in the range of 0.2 - 0.8. Angle less than 0.2 has quickly increasing
        computation time and angle greater 0.8 has quickly increasing error.

    n_jobs : int, default=None
        The number of parallel jobs to run for neighbors search. This parameter
        has no impact when ``metric="precomputed"`` or
        (``metric="euclidean"`` and ``method="exact"``).
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.22

    n_iter : int
        Maximum number of iterations for the optimization. Should be at
        least 250.

        .. deprecated:: 1.5
            `n_iter` was deprecated in version 1.5 and will be removed in 1.7.
            Please use `max_iter` instead.

    Attributes
    ----------
    embedding_ : array-like of shape (n_samples, n_components)
        Stores the embedding vectors.

    kl_divergence_ : float
        Kullback-Leibler divergence after optimization.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    learning_rate_ : float
        Effective learning rate.

        .. versionadded:: 1.2

    n_iter_ : int
        Number of iterations run.

    See Also
    --------
    sklearn.decomposition.PCA : Principal component analysis that is a linear
        dimensionality reduction method.
    sklearn.decomposition.KernelPCA : Non-linear dimensionality reduction using
        kernels and PCA.
    MDS : Manifold learning using multidimensional scaling.
    Isomap : Manifold learning based on Isometric Mapping.
    LocallyLinearEmbedding : Manifold learning using Locally Linear Embedding.
    SpectralEmbedding : Spectral embedding for non-linear dimensionality.

    Notes
    -----
    For an example of using :class:`~sklearn.manifold.TSNE` in combination with
    :class:`~sklearn.neighbors.KNeighborsTransformer` see
    :ref:`sphx_glr_auto_examples_neighbors_approximate_nearest_neighbors.py`.

    References
    ----------

    [1] van der Maaten, L.J.P.; Hinton, G.E. Visualizing High-Dimensional Data
        Using t-SNE. Journal of Machine Learning Research 9:2579-2605, 2008.

    [2] van der Maaten, L.J.P. t-Distributed Stochastic Neighbor Embedding
        https://lvdmaaten.github.io/tsne/

    [3] L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms.
        Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
        https://lvdmaaten.github.io/publications/papers/JMLR_2014.pdf

    [4] Belkina, A. C., Ciccolella, C. O., Anno, R., Halpert, R., Spidlen, J.,
        & Snyder-Cappione, J. E. (2019). Automated optimized parameters for
        T-distributed stochastic neighbor embedding improve visualization
        and analysis of large datasets. Nature Communications, 10(1), 1-12.

    [5] Kobak, D., & Berens, P. (2019). The art of using t-SNE for single-cell
        transcriptomics. Nature Communications, 10(1), 1-14.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.manifold import TSNE
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> X_embedded = TSNE(n_components=2, learning_rate='auto',
    ...                   init='random', perplexity=3).fit_transform(X)
    >>> X_embedded.shape
    (4, 2)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "perplexity": [Interval(Real, 0, None, closed="neither")],
        "learning_rate": [
            StrOptions({"auto"}),
            Interval(Real, 0, None, closed="neither"),
        ],
        "max_iter": [Interval(Integral, 250, None, closed="left"), None],
        "n_iter_without_progress": [Interval(Integral, -1, None, closed="left")],
        "min_grad_norm": [Interval(Real, 0, None, closed="left")],
        "metric": [StrOptions(set(_VALID_METRICS) | {"precomputed"}), callable],
        "metric_params": [dict, None],
        "init": [
            StrOptions({"sc", "pca", "random"}),
            np.ndarray,
        ],
        "verbose": ["verbose"],
        "random_state": ["random_state"],
        "angle": [Interval(Real, 0, 1, closed="both")],
        "n_jobs": [None, Integral],
        "n_iter": [
            Interval(Integral, 250, None, closed="left"),
            Hidden(StrOptions({"deprecated"})),
        ],
        "knn_graph": [Interval(Integral, 2, None, closed="left")],
        "gl_kernel": [StrOptions({"entropyperp", "gaussian", "uniform"})],
        "gl_normalization": [StrOptions({"combinatorial", "normalized"})],
        ##### PUT graph construction parameter option here
        "num_landmarks" : [None, Integral],
        "rpchol_iter":[None, Integral],
        "k_eigen":[None, Integral]
    }

    # Control the number of iterations (TO BE CHANGED POSSIBLY)
    _MAX_ITER = 250

    # Control the number of iterations between progress checks
    _N_ITER_CHECK = 50

    def __init__(
        self,
        n_components=2,
        *,
        perplexity=30.0,
        learning_rate="auto",
        max_iter=None,  # TODO(1.7): set to 1000
        n_iter_without_progress=300,
        min_grad_norm=1e-7,
        metric="euclidean",
        metric_params=None,
        init="sc",
        verbose=0,
        random_state=None,
        angle=0.5,
        n_jobs=None,
        n_iter="deprecated",
        knn_graph=20,
        gl_kernel="entropyperp",
        gl_normalization="combinatorial",
        rpchol_iter=50, 
        num_landmarks = 50,
        k_eigen = 50
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.n_iter_without_progress = n_iter_without_progress
        self.min_grad_norm = min_grad_norm
        self.metric = metric
        self.metric_params = metric_params
        self.init = init
        self.verbose = verbose
        self.random_state = random_state
        self.angle = angle
        self.n_jobs = n_jobs
        self.n_iter = n_iter
        self.knn_graph = knn_graph
        self.gl_kernel = gl_kernel
        self.gl_normalization = gl_normalization
        self.num_landmarks = num_landmarks
        self.k_eigen = k_eigen


    def _check_params_vs_input(self, X):
        if self.perplexity >= X.shape[0]:
            raise ValueError("perplexity must be less than n_samples")

    def _fit(self, X, skip_num_points=0):
        """Private function to fit the model using X as training data."""

        if isinstance(self.init, str) and self.init == "pca" and issparse(X):
            raise TypeError(
                "PCA initialization is currently not supported "
                "with the sparse input matrix. Use "
                'init="random" instead.'
            )

        if self.learning_rate == "auto":
            # See issue #18018
            self.learning_rate_ = X.shape[0] / 4   # assuming early exaggeration = 1.0
            self.learning_rate_ = np.maximum(self.learning_rate_, 50)
        else:
            self.learning_rate_ = self.learning_rate

        X = self._validate_data(
                X, accept_sparse=["csr", "csc", "coo"], dtype=[np.float32, np.float64]
            )
        
        random_state = check_random_state(self.random_state)

        n_samples = X.shape[0]

        # compute graph laplacian with graphlearning package
        if self.verbose:
            print("[t-SNE] Computing the graph Laplacian via GraphLearning package...")
        

        if self.gl_kernel in ["uniform", "gaussian"]:
            W = gl.weightmatrix.knn(X, self.knn_graph, kernel=self.gl_kernel)
        elif self.gl_kernel == "entropyperp":
            #############################################################
            #############################################################
            ########### ADD FASTER METHODS, leveraging Adam's Code #####
            ###########################################################
            ############################################################

            # compute k-nearest neighbors data
            knn_ind, knn_dist = gl.weightmatrix.knnsearch(X, self.knn_graph)
            D = knn_dist * knn_dist  # squared distances
            n = X.shape[0]

            if self.perplexity > self.knn_graph:
                print(f"perplexity {self.perplexity} is too large... setting to half of knn = {self.knn_graph}")
                self.perplexity = 0.5*self.knn_graph

            # prep for find entropy-perplexity smoothing of nearest neighbors  
            OFS = np.log(self.perplexity)
            similarity = lambda v, x: np.exp(-x*v)
            def entropy_obj(x, vals):
                # vals = vals/vals.max()
                probs = similarity(vals,x)
                probs /= probs.sum()
                mask = probs >= 1e-50
                # print(probs.min(), probs.max())
                return -1.0*(probs[mask]*np.log(probs[mask])).sum() - OFS

            # parallelize finding each of the bandwidths
            D /= D.max(axis=1).reshape(-1,1)
            D = D * D
            gammas2 = Parallel(n_jobs=self.n_jobs)(delayed(opt_bisection)(partial(entropy_obj, vals=D[i, 1:])) for i in range(knn_dist.shape[0]))
            weights = D * np.array(gammas2).reshape(-1,1)   # apply the entropy-scaled bandwidths

            ##### Finish construction of weightmatrix (copy-pasted from graphlearning.weightmatrix.knn)
            #Flatten knn data and weights
            knn_ind = knn_ind.flatten()
            weights = weights.flatten()

            #Self indices
            self_ind = np.ones((n,self.knn_graph))*np.arange(n)[:,None]
            self_ind = self_ind.flatten()

            #Construct sparse matrix and convert to Compressed Sparse Row (CSR) format
            W = coo_matrix((weights, (self_ind, knn_ind)),shape=(n,n)).tocsr()
            W = (W + W.transpose()) / 2.0
            W.setdiag(0)

        self.Graph = gl.graph(W)
        L = self.Graph.laplacian(normalization=self.gl_normalization)
        if self.verbose >= 2:
            print(f"Graph is connected = {self.Graph.isconnected()}")
        

        #######################################################
        #####################################################
        ######### CHANGE INITIALIZATION METHODS HERE #########
        ####### rename sc, use similar to Adam's code?
        ###############################################
        if isinstance(self.init, np.ndarray):
            X_embedded = self.init
        elif self.init == "pca":
            pca = PCA(
                n_components=self.n_components,
                svd_solver="randomized",
                random_state=random_state,
            )
            # Always output a numpy array, no matter what is configured globally
            pca.set_output(transform="default")
            X_embedded = pca.fit_transform(X).astype(np.float32, copy=False)
            # PCA is rescaled so that PC1 has standard deviation 1e-4 which is
            # the default value for random initialization. See issue #18018.
            X_embedded = X_embedded / np.std(X_embedded[:, 0]) * 1e-4
        elif self.init == "random":
            # The embedding is initialized with iid samples from Gaussians with
            # standard deviation 1e-4.
            X_embedded = 1e-4 * random_state.standard_normal(
                size=(n_samples, self.n_components)
            ).astype(np.float32)
        elif self.init == "sc": # Spectral Clustering initialization
            _, X_embedded = self.Graph.eigen_decomp(k=self.n_components+1) 
            X_embedded = X_embedded[:,1:].astype(np.float32)

        if self.gl_kernel == "qij":
            knn_inds = W.nonzero()
            qdata = 1./ (1. + np.linalg.norm(X_embedded[knn_inds[0]] - X_embedded[knn_inds[1]], axis=1)**2.)
            W[knn_inds] = qdata / qdata.sum()
            self.Graph = gl.graph(W)
            L = self.Graph.laplacian(normalization=self.gl_normalization)
            if self.verbose >= 2:
                print(f"Graph (with Q_ij) is connected = {self.Graph.isconnected()}")

        del W # don't need this stored anymore

        self.V, evals = self.Graph.eigen_decomp(k=self.k_eigen+1)
        self.V, evals = self.V[:,1:], evals[1:]
        A_embedded = self.V.T @ X_embedded    # project onto eigenfunctions

        return self._lap_tsne_eig(
            evals,
            self.k_eigen,
            A_embedded=A_embedded,
            skip_num_points=skip_num_points,
        )

    def _lap_tsne_eig(
        self,
        evals,
        n_samples,
        A_embedded,
        neighbors=None,
        skip_num_points=0,
    ):
        """Runs Laplacian approximation of t-SNE."""

        params = A_embedded.ravel()
        
        opt_args = {
            "it": 0,
            "n_iter_check": self._N_ITER_CHECK,
            "min_grad_norm": self.min_grad_norm,
            "learning_rate": self.learning_rate_,
            "verbose": self.verbose,
            "kwargs": dict(skip_num_points=skip_num_points),
            "args": [evals, n_samples, self.n_components],
            "n_iter_without_progress": self._MAX_ITER,
            "max_iter": self._MAX_ITER,
            "momentum": 0.5,
        }

        opt_args["num_landmarks"] = self.num_landmarks
        obj_func = _laplacian_kl_divergence_eigen

        # no early exaggeration, and so we start optimization process with the higher momentum at 0.8
        remaining = self._max_iter
        opt_args["max_iter"] = self._max_iter
        opt_args["it"] = 0
        opt_args["momentum"] = 0.8
        opt_args["n_iter_without_progress"] = self.n_iter_without_progress

        if self.verbose >= 2:
            params, kl_divergence, it, iterates = _gradient_descent(obj_func, params, **opt_args)
            self.iterates = [x.reshape(n_samples, self.n_components) for x in iterates]
        else:
            params, kl_divergence, it = _gradient_descent(obj_func, params, **opt_args)

        # Save the final number of iterations
        self.n_iter_ = it

        if self.verbose:
            extra_str = f"(Laplacian-based approx)"
            print(
                "[t-SNE] KL divergence %s after %d iterations: %f"
                % (extra_str, it + 1, kl_divergence)
            )


        
        self.A = params.reshape(self.k_eigen, self.n_components) # save the learned eigenfunction coeffs as member property
        X_embedded = self.V @ self.A # return the full embedding from the eigenfunction coeffs 
        self.kl_divergence_ = kl_divergence

        return X_embedded
    



    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit_transform(self, X, y=None):
        """Fit X into an embedded space and return that transformed output.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of the training data in low-dimensional space.
        """
        # TODO(1.7): remove
        # Also make sure to change `max_iter` default back to 1000 and deprecate None
        if self.n_iter != "deprecated":
            if self.max_iter is not None:
                raise ValueError(
                    "Both 'n_iter' and 'max_iter' attributes were set. Attribute"
                    " 'n_iter' was deprecated in version 1.5 and will be removed in"
                    " 1.7. To avoid this error, only set the 'max_iter' attribute."
                )
            warnings.warn(
                (
                    "'n_iter' was renamed to 'max_iter' in version 1.5 and "
                    "will be removed in 1.7."
                ),
                FutureWarning,
            )
            self._max_iter = self.n_iter
        elif self.max_iter is None:
            self._max_iter = 1000
        else:
            self._max_iter = self.max_iter

        self._check_params_vs_input(X)
        embedding = self._fit(X)
        self.embedding_ = embedding
        return self.embedding_

    @_fit_context(
        # TSNE.metric is not validated yet
        prefer_skip_nested_validation=False
    )
    def fit(self, X, y=None):
        """Fit X into an embedded space.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features) or \
            (n_samples, n_samples)
            If the metric is 'precomputed' X must be a square distance
            matrix. Otherwise it contains a sample per row. If the method
            is 'exact', X may be a sparse matrix of type 'csr', 'csc'
            or 'coo'. If the method is 'barnes_hut' and the metric is
            'precomputed', X may be a precomputed sparse graph.

        y : None
            Ignored.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.fit_transform(X)
        return self

    @property
    def _n_features_out(self):
        """Number of transformed output features."""
        return self.embedding_.shape[1]

    def _more_tags(self):
        return {"pairwise": self.metric == "precomputed"}
