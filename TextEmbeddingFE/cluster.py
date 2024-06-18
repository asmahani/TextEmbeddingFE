import numpy as np
from sklearn.utils.validation import check_array

class SphericalKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.inertia_ = None

    def fit(self, X, y = None):
        X = check_array(X)
        X = self._normalize(X)
        
        best_inertia = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # Initialize centroids
            self.centroids = self._initialize_centroids(X)
            
            iter_count = 0
            while iter_count < self.max_iter:
                try:
                    # Assign clusters
                    self.labels_ = self._assign_clusters(X, hard=True)
                    # Compute new centroids
                    new_centroids = self._compute_centroids(X)
                    # Check for convergence
                    if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                        break
                    self.centroids = new_centroids
                    iter_count += 1
                except Exception as e:
                    # Handle any error during the computation of centroids
                    print(f"Error encountered: {e}. Resetting centroids and continuing.")
                    iter_count = 0
                    self.centroids = self._initialize_centroids(X)
                    continue

            # Calculate total within-cluster inertia
            inertia = self._calculate_inertia(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = self.centroids
                best_labels = self.labels_

        # Set the best results
        self.cluster_centers_ = best_centroids
        self.labels_ = best_labels
        self.inertia_ = best_inertia

        return self

    def predict(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=True)
    
    def transform(self, X):
        X = check_array(X)
        X = self._normalize(X)
        return self._assign_clusters(X, hard=False)
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _initialize_centroids(self, X):
        # Randomly select n_clusters points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, hard=True):
        # Calculate the cosine similarity between each point and each centroid
        #similarities = np.array([[self._calculate_similarity(x, centroid) for centroid in self.centroids] for x in X])
        similarities = np.dot(X, self.centroids.T)
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

    def _calculate_similarity(self, x1, x2):
        # Cosine similarity
        return np.dot(x1, x2)# / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _normalize(self, X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / norms

    def _calculate_inertia(self, X):
        total_inertia = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            centroid = self.centroids[k]
            distances = [1 - self._calculate_similarity(point, centroid) for point in cluster_points]
            total_inertia += np.sum(distances)
        return total_inertia

    def fit_predict(self, X, y = None):
        self.fit(X)
        return self.labels_
