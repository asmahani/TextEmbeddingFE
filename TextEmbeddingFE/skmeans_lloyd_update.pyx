# distutils: language=c++

cdef extern from "skmeans_lloyd_update_cpp.cpp":
    void skmeans_lloyd_update_cpp(double* X, double* centroids, double* similarities, int n_obs, int n_clusters, int n_features, 
                              int* new_labels, double* new_centroids, double* frobenius_norm)

import numpy as np
cimport numpy as np

def skmeans_lloyd_update_v2(np.ndarray[np.float64_t, ndim=2] X,
                         np.ndarray[np.float64_t, ndim=2] centroids,
                         np.ndarray[np.float64_t, ndim=2] similarities):
    cdef int n_obs = X.shape[0]
    cdef int n_clusters = centroids.shape[0]
    cdef int n_features = X.shape[1]
    
    cdef np.ndarray[np.float64_t, ndim=2] new_centroids = np.zeros((n_clusters, n_features), dtype=np.float64)
    cdef np.ndarray[np.int_t, ndim=1] new_labels = np.empty(n_obs, dtype=np.int)
    cdef double frobenius_norm
    
    skmeans_lloyd_update_cpp(&X[0, 0], &centroids[0, 0], &similarities[0, 0], 
                         n_obs, n_clusters, n_features, 
                         <int*>new_labels.data, &new_centroids[0, 0], &frobenius_norm)
    
    return similarities, new_labels, new_centroids, frobenius_norm
