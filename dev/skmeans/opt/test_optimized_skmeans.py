import pandas as pd

import TextEmbeddingFE.cluster
import TextEmbeddingFE.main

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 3072

X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()

from TextEmbeddingFE.cluster import SphericalKMeans as SKMeans1
from TextEmbeddingFE.cluster2 import SphericalKMeans as SKMeans2

import time

my_n_init = 10

t1 = time.time()
obj1 = SKMeans1(n_clusters = my_n_cluster, n_init = my_n_init).fit(X)
t1 = time.time() - t1
print(f'time - v1: {round(t1, 2)}sec')

t2 = time.time()
obj2 = SKMeans2(n_clusters = my_n_cluster, n_init = my_n_init).fit(X)
t2 = time.time() - t2
print(f'time - v2: {round(t2, 2)}sec')

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
ari = adjusted_rand_score(obj1.labels_, obj2.labels_)
ami = adjusted_mutual_info_score(obj1.labels_, obj2.labels_)
print(f'ari: {ari}, ami: {ami}')

