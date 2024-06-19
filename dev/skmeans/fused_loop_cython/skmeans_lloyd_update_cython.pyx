# skmeans_lloyd_update.pyx
import numpy as np
cimport numpy as np
from libc.math cimport sqrt, INFINITY

def skmeans_lloyd_update_v2(np.ndarray[np.float64_t, ndim=2] X, np.ndarray[np.float64_t, ndim=2] centroids):
    cdef int n_obs = X.shape[0]
    cdef int n_clusters = centroids.shape[0]
    cdef int n_features = X.shape[1]
    
    cdef np.ndarray[np.int_t, ndim=1] new_labels = np.empty(n_obs, dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] new_centroids = np.zeros_like(centroids)
    cdef np.ndarray[np.float64_t, ndim=2] similarities = np.zeros((n_obs, n_clusters))
    cdef double frobenius_norm = 0.0
    cdef double max_similarity, similarity
    cdef int n, k, best_cluster

    for n in range(n_obs):
        max_similarity = -INFINITY
        best_cluster = -1

        for k in range(n_clusters):
            similarity = np.dot(X[n, :], centroids[k, :])
            similarities[n, k] = similarity
            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster = k

        new_labels[n] = best_cluster
        new_centroids[best_cluster, :] += X[n, :]

    for k in range(n_clusters):
        norm = np.linalg.norm(new_centroids[k, :])
        if norm == 0:
            raise RuntimeError('One or more clusters are empty')
        
        new_centroids[k, :] /= norm
        frobenius_norm += np.sum((new_centroids[k, :] - centroids[k, :]) ** 2)

    frobenius_norm = sqrt(frobenius_norm)

    return similarities, new_labels, new_centroids, frobenius_norm
