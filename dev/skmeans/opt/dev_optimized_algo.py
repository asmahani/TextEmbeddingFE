import numpy as np

def skmeans_lloyd_update(
        X
        , centroids
):
    n_clusters, n_features = centroids.shape
    return n_clusters, n_features

####

import pandas as pd

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 5
X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()
init_centroids = X[np.random.choice(X.shape[0], size = my_n_cluster, replace = False), :]

skmeans_lloyd_update(X, init_centroids)
