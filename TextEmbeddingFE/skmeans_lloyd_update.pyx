# cython: boundscheck=False, wraparound=False
import numpy as np
cimport numpy as np

def skmeans_lloyd_update(
        np.ndarray[double, ndim=2] X,
        np.ndarray[double, ndim=2] centroids,
        np.ndarray[double, ndim=2] similarities):
    
    cdef int n_obs = X.shape[0]
    cdef int n_clusters = centroids.shape[0]
    cdef int n_features = centroids.shape[1]
    
    cdef np.ndarray[int, ndim=1] new_labels = np.empty(n_obs, dtype=np.int32)
    cdef np.ndarray[double, ndim=2] new_centroids = np.zeros_like(centroids)
    cdef int n
    cdef int k
    cdef double max_sim
    cdef int max_idx

    for n in range(n_obs):
        max_sim = -1.0
        max_idx = -1
        for k in range(n_clusters):
            if similarities[n, k] > max_sim:
                max_sim = similarities[n, k]
                max_idx = k
        new_labels[n] = max_idx
        for k in range(n_features):
            new_centroids[max_idx, k] += X[n, k]
    
    cdef np.ndarray[double, ndim=2] my_norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
    
    if np.any(my_norms == 0):
        raise RuntimeError('One or more clusters are empty')

    for k in range(n_clusters):
        for f in range(n_features):
            new_centroids[k, f] /= my_norms[k, 0]
    
    cdef np.ndarray[double, ndim=2] new_similarities = np.dot(X, new_centroids.T)
    
    cdef double frobenius_norm = 0.0
    for n in range(n_clusters):
        for f in range(n_features):
            frobenius_norm += (new_centroids[n, f] - centroids[n, f]) ** 2
    
    frobenius_norm = np.sqrt(frobenius_norm)
    
    return new_similarities, new_labels, new_centroids, frobenius_norm
