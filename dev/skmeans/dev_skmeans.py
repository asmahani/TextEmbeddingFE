import numpy as np
from sklearn.utils.validation import check_array

import warnings

class SphericalKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.total_within_cluster_distance_ = None

    def fit(self, X, y = None):
        X = check_array(X)
        
        best_total_within_cluster_distance = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # Normalize the data
            X = self._normalize(X)
            
            # Initialize centroids
            self.centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iter):
                # Assign clusters
                self.labels_ = self._assign_clusters(X, hard = True)
                # Compute new centroids
                new_centroids = self._compute_centroids(X)
                # Check for convergence
                if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                    break
                self.centroids = new_centroids

            # Calculate total within-cluster distance
            total_within_cluster_distance = self._calculate_total_within_cluster_distance(X)
            
            if total_within_cluster_distance < best_total_within_cluster_distance:
                best_total_within_cluster_distance = total_within_cluster_distance
                best_centroids = self.centroids
                best_labels = self.labels_

        # Set the best results
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.total_within_cluster_distance_ = best_total_within_cluster_distance

        return self

    def predict(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard = True)
    
    def transform(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard = False)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _initialize_centroids(self, X):
        # Randomly select n_clusters points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, hard = True):
        # Calculate the cosine similarity between each point and each centroid
        similarities = np.array([[self._calculate_similarity(x, centroid) for centroid in self.centroids] for x in X])
        # Assign each point to the nearest centroid (highest similarity)
        if hard:
            return np.argmax(similarities, axis=1)
        else:
            return similarities

    def _compute_centroids(self, X):
        # Compute the mean of the points in each cluster to find the new centroids
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                centroids[k] = self._normalize(cluster_points.mean(axis=0).reshape(1, -1))
            else:
                raise RuntimeError('one or more cluster(s) have zero members')
        return centroids

    def _calculate_similarity_debug(self, x1, x2):
        # Cosine similarity
        with warnings.catch_warnings(record = True) as w:
            norm1, norm2 = np.linalg.norm(x1), np.linalg.norm(x2)
            if w:
                print(norm1)
                print(norm2)
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return np.dot(x1, x2) / (norm1 * norm2)

    def _calculate_similarity(self, x1, x2):
        # Cosine similarity
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    
    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("Zero vector encountered during normalization")
        return X / norms

    def _calculate_total_within_cluster_distance(self, X):
        total_distance = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            centroid = self.centroids[k]
            distances = [1 - self._calculate_similarity(point, centroid) for point in cluster_points]
            total_distance += np.sum(distances)
        return total_distance

    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.labels_

#####

import pandas as pd

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

X = dfEmbeddings.iloc[:, 2:7]
X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X)

obj_skmeans = SphericalKMeans().fit(X)

np.apply_along_axis(lambda x: np.sum(x*x), 1, obj_skmeans.centroids)

hard_labels_skmeans = obj_skmeans.predict(X)
soft_labels_skmeans = obj_skmeans.transform(X)

from sklearn.cluster import KMeans

obj_kmeans = KMeans(n_clusters=3).fit(X)
hard_labels_kmeans = obj_kmeans.predict(X)
soft_labels_kmeans = obj_kmeans.transform(X)

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score

ari = adjusted_rand_score(hard_labels_kmeans, hard_labels_skmeans)
ami = adjusted_mutual_info_score(hard_labels_kmeans, hard_labels_skmeans)

print(f'ari: {ari}, ami: {ami}')