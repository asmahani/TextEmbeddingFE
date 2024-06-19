import numpy as np
from TextEmbeddingFE.cluster import skmeans_lloyd_update

def skmeans_lloyd_update_v2(X, centroids):
    n_obs, n_features = X.shape
    n_clusters = centroids.shape[0]

    new_labels = np.empty(n_obs, dtype=int)
    new_centroids = np.zeros_like(centroids)
    similarities = np.zeros((n_obs, n_clusters))
    frobenius_norm = 0.0

    for n in range(n_obs):
        max_similarity = -np.inf
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

    frobenius_norm = np.sqrt(frobenius_norm)

    return similarities, new_labels, new_centroids, frobenius_norm

def skmeans_lloyd_update_v3(X, centroids):
    n_obs, n_features = X.shape
    n_clusters = centroids.shape[0]

    new_labels = np.empty(n_obs, dtype=int)
    new_centroids = np.zeros_like(centroids)
    similarities = np.zeros((n_obs, n_clusters))
    frobenius_norm = 0.0

    for n in range(n_obs):
        max_similarity = -np.inf
        best_cluster = -1

        for k in range(n_clusters):
            similarity = np.dot(X[n, :], centroids[k, :])
            similarities[n, k] = similarity
            if similarity > max_similarity:
                max_similarity = similarity
                best_cluster = k

        new_labels[n] = best_cluster
        new_centroids[best_cluster, :] += X[n, :]

    my_norms = np.linalg.norm(new_centroids, axis = 1).reshape(-1, 1)
    if np.any(my_norms == 0):
        raise RuntimeError('One or more clusters are empty')

    new_centroids = new_centroids / my_norms

    frobenius_norm = np.sqrt(np.sum((new_centroids - centroids) ** 2))

    return similarities, new_labels, new_centroids, frobenius_norm

# Example usage:
X = np.random.rand(100, 3)  # 100 observations with 3 features
X = X / np.linalg.norm(X, axis = 1).reshape(-1, 1)
centroids = X[np.random.choice(X.shape[0], 5, replace = False), :]

#%timeit
similarities, new_labels, new_centroids, frobenius_norm = skmeans_lloyd_update_v3(X, centroids)

#%timeit
new_similarities, new_labels_v2, new_centroids_v2, frobenius_norm_v2 = skmeans_lloyd_update(X, centroids, similarities)

assert np.isclose(frobenius_norm, frobenius_norm_v2)
assert np.allclose(new_labels, new_labels_v2)
assert np.allclose(new_centroids, new_centroids_v2)
assert np.array_equal(new_labels, new_labels_v2)
