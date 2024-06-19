import pandas as pd

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 10
my_n_features = 1000

X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()

from TextEmbeddingFE.cluster import SphericalKMeans as SKMeans
from sklearn.cluster import KMeans

def test_kmean():
    KMeans(n_clusters = my_n_cluster).fit(X)
    pass

%timeit test_kmean()

def test_skmean():
    SKMeans(n_clusters = my_n_cluster).fit(X)
    pass

%timeit test_skmean()

