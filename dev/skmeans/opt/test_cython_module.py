import pandas as pd

import TextEmbeddingFE.cluster
import TextEmbeddingFE.main

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 3072

X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()

import numpy as np

X_unique = np.unique(X, axis = 0)
init_indices = np.random.choice(X_unique.shape[0], size = my_n_cluster, replace = False)
init_centroids = X_unique[init_indices, :]
init_similarities = np.dot(X, init_centroids.T)

from TextEmbeddingFE.cluster2 import skmeans_lloyd_update
from TextEmbeddingFE.skmeans_lloyd_update import skmeans_lloyd_update as cythonFunction

import time

nrep = 100

t1 = time.time()
for n in range(nrep):
    new_similarities, new_labels, new_centroids, frobenius_norm = skmeans_lloyd_update(
        X = X, centroids = init_centroids, similarities = init_similarities
    )
t1 = time.time() - t1
print(f'time - python: {round(t1, 2)}sec')

t2 = time.time()
for n in range(nrep):
    new_similarities_2, new_labels_2, new_centroids_2, frobenius_norm_2 = cythonFunction(
        X = X, centroids = init_centroids, similarities = init_similarities
    )
t2 = time.time() - t2
print(f'time - Cython: {round(t2, 2)}sec')

