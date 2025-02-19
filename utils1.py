import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from scipy.sparse import coo_matrix
import numpy as np
import scipy.sparse as sp
from numba import njit, prange
from sklearn.metrics.pairwise import pairwise_distances

tab10 = plt.cm.tab10.colors


def make_neon(color, saturation_factor=1.2, brightness_factor=1.1):
    # Convert RGB to HSV
    h, s, v = mcolors.rgb_to_hsv(color)
    
    # Increase saturation and brightness
    s = min(1.0, s * saturation_factor)
    v = min(1.0, v * brightness_factor)
    
    # Convert back to RGB
    return mcolors.hsv_to_rgb((h, s, v))

neon_tab10 = [make_neon(color) for color in tab10]
from matplotlib.colors import ListedColormap

# Create a ListedColormap from the neon_tab10 colors
neon_tab10_cmap = ListedColormap(neon_tab10)
neon_tab10[1] = (1.0, 0.4980392156862745, 0.054901960784313725)
neon_tab10[8] = (0.94, 0.88, 0.19)
neon_tab10_cmap = ListedColormap(neon_tab10)


def UMAP_graph_weight_perplexity1(knn_graph,sigma):
    n = knn_graph.shape[0]
    knn_distances = knn_graph.data.reshape(n,-1)
    rho = knn_distances.min(axis=1)
    normalized_distances = knn_distances-rho[:,np.newaxis]
    normalized_distances/=sigma[:,np.newaxis]
    wij = np.exp(-normalized_distances**2)
    return np.sum(wij,axis=1)


def UMAP_graph_weights1(knn_graph,sigma):
    n = knn_graph.shape[0]
    knn_distances = knn_graph.data.reshape(n,-1)
    rho = knn_distances.min(axis=1)
    normalized_distances = knn_distances-rho[:,np.newaxis]
    normalized_distances/=sigma[:,np.newaxis]
    wij = np.exp(-normalized_distances**2)
    return wij


def vectorized_binary_search1(knn_distances, targets, low, high, tol=1e-6, max_iter=100):
    low, high = np.array(low), np.array(high)  
    for _ in range(max_iter):
        mid = (low + high) / 2  
        vals = UMAP_graph_weight_perplexity1(knn_distances, mid) 
        errors = np.abs(vals - targets) 
        mask = vals > targets 
        low = np.where(mask, low, mid) 
        high = np.where(mask, mid, high) 
        if np.all(errors < tol):
            break
    return mid


def UMAP_graph1(knn_graph, k):
    n = knn_graph.shape[0]
    knn_graph.setdiag(0)
    knn_graph.eliminate_zeros()
    rows, cols = knn_graph.nonzero()
    targets = np.log2(k)*np.ones(n)
    low = np.zeros(n)
    high = 100000*np.ones(n)
    sigma = vectorized_binary_search1(knn_graph, targets, low, high)
    weights = UMAP_graph_weights1(knn_graph,sigma).flatten()
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    symmetrized_graph = graph+graph.T-graph.multiply(graph.T)
    return symmetrized_graph




def UMAP_graph_weight_perplexity(knn_graph, sigma):
    # Ensure knn_graph is in CSR format for efficient row slicing
    knn_graph = knn_graph.tocsr()
    n = knn_graph.shape[0]
    results = np.empty(n, dtype=float)
    
    # Loop over each row in the graph
    for i in range(n):
        # Extract distances (nonzero entries) for row i
        row_data = knn_graph.getrow(i).data
        if row_data.size > 0:
            # Compute rho as the minimum distance in the row
            rho = row_data.min()
            # Normalize the distances (broadcasting works even if row_data is 1D)
            normalized = (row_data - rho) / sigma[i]
            # Compute weights using the exponential decay
            wij = np.exp(-normalized)
            # Sum up the weights
            results[i] = wij.sum()
        else:
            # If the row is empty, set result to 0 (or handle as needed)
            results[i] = 0.0

    return results





def UMAP_graph_weights(knn_graph, sigma):
    """
    Compute the UMAP weights for each edge in the knn_graph.
    
    Parameters
    ----------
    knn_graph : scipy.sparse matrix
        A symmetric kNN graph (e.g. in CSR format) where each row contains 
        distances to neighbors. Different rows may have a different number 
        of nonzero entries.
    sigma : array_like, shape (n,)
        The sigma (scaling) values per row.
    
    Returns
    -------
    wij_list : list of 1D numpy arrays
        A list where each element is a 1D array of weights corresponding to the 
        nonzero entries of the respective row in knn_graph.
    """
    # Ensure the graph is in CSR format for efficient row slicing.
    knn_graph = knn_graph.tocsr()
    n = knn_graph.shape[0]
    
    wij_list = []
    for i in range(n):
        # Extract the distances for row i.
        row_data = knn_graph.getrow(i).data
        if row_data.size > 0:
            # Compute rho as the minimum distance in this row.
            rho = row_data.min()
            # Normalize the distances by subtracting rho and dividing by sigma[i].
            normalized = (row_data - rho) / sigma[i]
            # Compute weights using the exponential decay.
            weights = np.exp(-normalized)
        else:
            weights = np.array([])  # If no neighbors, return an empty array.
        wij_list.append(weights)
    
    return np.concatenate(wij_list)


def vectorized_binary_search(knn_distances, targets, low, high, tol=1e-6, max_iter=100):
    low, high = np.array(low), np.array(high)  
    for _ in range(max_iter):
        mid = (low + high) / 2  
        vals = UMAP_graph_weight_perplexity(knn_distances, mid) 
        errors = np.abs(vals - targets) 
        mask = vals > targets 
        low = np.where(mask, low, mid) 
        high = np.where(mask, mid, high) 
        if np.all(errors < tol):
            break
    return mid


def UMAP_graph(knn_graph, k):
    n = knn_graph.shape[0]
    knn_graph.setdiag(0)
    knn_graph.eliminate_zeros()
    rows, cols = knn_graph.nonzero()
    targets = np.log2(k)*np.ones(n)
    low = np.zeros(n)
    high = 100000*np.ones(n)
    sigma = vectorized_binary_search(knn_graph, targets, low, high)
    weights = UMAP_graph_weights(knn_graph,sigma)#.flatten()
    graph = csr_matrix((weights, (rows, cols)), shape=(n, n))
    symmetrized_graph = graph+graph.T-graph.multiply(graph.T)
    return symmetrized_graph

def repulsive_force_A_with_subsample_singular_without_renorm(A, n, l_points, vecs,P, a = 1):
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
    grad = (2.0) * (S2 @ Y - S2.sum(axis=1).reshape(-1,1) * Y)

    return vecs[l, :].T@grad