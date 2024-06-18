import pandas as pd

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 3072

X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()

from TextEmbeddingFE.cluster import SphericalKMeans as SKMeans

import time

my_n_init = 10

t1 = time.time()
obj1 = SKMeans(n_clusters = my_n_cluster, n_init = my_n_init).fit(X)
t1 = time.time() - t1
print(f'time - v1: {round(t1, 2)}sec')

