import numpy as np
#from TextEmbeddingFE.cluster import skmeans_lloyd_update as skmeans_lloyd_update_original
from skmeans_lloyd_update_cython import skmeans_lloyd_update_v2 as skmeans_lloyd_update_opt_cython

def skmeans_lloyd_update_opt_python(X, centroids):
    n_obs, n_features = X.shape
    n_clusters = centroids.shape[0]

    new_labels = np.empty(n_obs, dtype=int)
    new_centroids = np.zeros_like(centroids)
    #similarities = np.zeros((n_obs, n_clusters))
    frobenius_norm = 0.0

    for n in range(n_obs):
        max_similarity = -np.inf
        best_cluster = -1

        for k in range(n_clusters):
            similarity = np.dot(X[n, :], centroids[k, :])
            #similarities[n, k] = similarity
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

    frobenius_norm = np.sqrt(frobenius_norm)

    #return similarities, new_labels, new_centroids, frobenius_norm
    return new_labels, new_centroids, frobenius_norm

# Example usage
X = np.random.rand(100, 3)  # 100 observations with 3 features
X = X / np.linalg.norm(X, axis=1).reshape(-1, 1)
centroids = X[np.random.choice(X.shape[0], 5, replace=False), :]

# Time the Python implementation
#%timeit 
#similarities_py, new_labels_py, new_centroids_py, frobenius_norm_py = skmeans_lloyd_update_opt_python(X, centroids)
%timeit new_labels_py, new_centroids_py, frobenius_norm_py = skmeans_lloyd_update_opt_python(X, centroids)

# Time the Cython implementation
%timeit similarities_cy, new_labels_cy, new_centroids_cy, frobenius_norm_cy = skmeans_lloyd_update_opt_cython(X, centroids)

# Verify the results are the same
if False:
    assert np.isclose(frobenius_norm_py, frobenius_norm_cy)
    assert np.allclose(new_labels_py, new_labels_cy)
    assert np.allclose(new_centroids_py, new_centroids_cy)
    assert np.array_equal(new_labels_py, new_labels_cy)
