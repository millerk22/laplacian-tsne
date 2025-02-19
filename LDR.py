import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml
import graphlearning as gl
from scipy.sparse import csr_matrix
from utils1 import UMAP_graph1, UMAP_graph


def load_data(dataset_name, PCA_flag=True, n_components=64):
    
    # Load dataset from OpenML
    dataset = fetch_openml(dataset_name, version=1)
    X, y = dataset.data, dataset.target
    y = y.astype(int)
    y = np.array(y)
    X = np.array(X)

    # Apply PCA if PCA_flag is True
    if PCA_flag:
        pca = PCA(n_components=n_components)
        X = pca.fit_transform(X)

    return X, y


def affinity_matrix(X: np.ndarray, k: int, t: int, p: int) -> csr_matrix:
    """
    Computes the affinity matrix using k-nearest neighbors and pruning based 
    on diffusion distances.

    Parameters:
    - X (np.ndarray): Dataset of points with shape (n_points, n_features),
                       where n_points is the number of data points and 
                       n_features is the dimensionality of each point.
    - k (int): The number of neighbors to consider for the k-NN graph.
    - t (int): The number of time steps to propagate the diffusion process.
    - p (int): The number eigenfunctions to retain.

    Returns:
    - csr_matrix: A sparse matrix reflecting the persistant neighbors defined
                  via the diffusion distances.
    """
    n = X.shape[0]
    knn_graph = gl.weightmatrix.knn(X, k, kernel='distance', symmetrize=False)
    P_matrix = UMAP_graph1(knn_graph, 30)
    with np.errstate(over='ignore'):
        P = gl.graph(P_matrix)
    
    vals_diffusion, vecs_diffusion = P.eigen_decomp(k = 200, normalization='normalized')
    dif_vals = 1-vals_diffusion
    Diffusion_coordinates = vecs_diffusion[:,1:].T*dif_vals[1:,None]**t
    W_diffused = gl.weightmatrix.knn(Diffusion_coordinates.T,k, kernel = 'uniform', symmetrize=False)
    retained_dist = W_diffused.multiply(knn_graph)
    
    dist = knn_graph.data.reshape(n,-1)
    first_nn_dist = np.min(dist, axis = 1)
    neighbor_indices = knn_graph.indices.reshape(n, -1)
    first_nn_pos = np.argmin(dist, axis=1)
    first_nn_actual = neighbor_indices[np.arange(n), first_nn_pos]
    rows = np.arange(n)
    first_nn_graph = csr_matrix((first_nn_dist, (rows, first_nn_actual)), shape=P_matrix.shape)
    
    new_distances = first_nn_graph.maximum(retained_dist)
    P_matrix_pruned = UMAP_graph(new_distances, 30)
    return P_matrix_pruned


def repulsive_force_A_with_subsample_singular(A, n, l_points, vecs,P, a = 0.015):
    l = np.random.choice(n, l_points, replace=False)
    Y = vecs[l, :] @ A
    squared_distances = pairwise_distances(Y, metric="euclidean", squared=True)
    repulsive_term = np.clip(-np.log(0.00000001+a*squared_distances**(1/2)),0,None)


    probs = np.ones((l_points,l_points))-P[l[:,None],l]

    
    np.fill_diagonal(repulsive_term, 0)

    repulsive_term*=probs
    
    repulsive_sum = repulsive_term.sum()
    distances = np.sqrt(squared_distances)
    S2 = np.zeros_like(distances)
    nonzero_mask = distances > 0
    valid_mask = nonzero_mask & (distances < 1 / a)
    S2[valid_mask] = 1 / squared_distances[valid_mask] 

    S2*=probs
    grad = (2.0/repulsive_sum) * (S2 @ Y - S2.sum(axis=1).reshape(-1,1) * Y)

    return vecs[l, :].T@grad