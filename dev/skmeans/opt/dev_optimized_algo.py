import numpy as np

def skmeans_lloyd_update(
        X
        , centroids
        , similarities
):
    n_clusters, n_features = centroids.shape
    n_obs = X.shape[0]

    new_labels = np.empty(n_obs, dtype = int)
    new_sizes = np.zeros(n_clusters, dtype = int)
    new_centroids = np.zeros_like(centroids)
    for n in range(n_obs):
        new_labels[n] = np.argmax(similarities[n, :])
        new_sizes[new_labels[n]] += 1
        new_centroids[new_labels[n], :] += X[n, :]
    
    if np.any(new_sizes == 0):
        raise RuntimeError('One or more clusters are empty')

    new_centroids = new_centroids / np.linalg.norm(new_centroids, axis = 1).reshape(-1, 1)
    new_similarities = np.dot(X, new_centroids.T)
    new_inertia = np.sum(1 - np.max(new_similarities, axis = 1))

    frobenius_norm = np.sqrt(np.sum((new_centroids - centroids) ** 2))

    return new_similarities, new_labels, new_sizes, new_centroids, new_inertia, frobenius_norm

####

import pandas as pd

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 3072

X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()
X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X)

import time

X_unique = np.unique(X, axis = 0)

t = time.time()

init_indices = np.random.choice(X_unique.shape[0], size = my_n_cluster, replace = False)
init_centroids = X_unique[init_indices, :]
init_similarities = np.dot(X, init_centroids.T)

my_similarities = init_similarities
my_centroids = init_centroids

for n in range(50):
    my_similarities, my_labels, my_sizes, my_centroids, my_inertia, my_delta = skmeans_lloyd_update(
        X, my_centroids
        , my_similarities
    )
    print(f'iter {n+1}, intertia: {my_inertia}, fb norm: {round(my_delta, 2)}')
    if my_delta < 1e-4:
        break
t = time.time() - t
print(f'time: {round(t, 2)}sec')



