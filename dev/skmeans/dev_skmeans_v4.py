import pandas as pd
from TextEmbeddingFE.cluster import SphericalKMeans

dfEmbeddings = pd.read_csv(
    'C:/Users/alire/OneDrive/data/statman_bitbucket/aki/LLM/March2024/openai_3large_operation.csv'
)

my_n_cluster = 5
my_n_features = 3072
X = dfEmbeddings.iloc[:, 2:(2 + my_n_features)].to_numpy()
#X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X)
obj_skmeans = SphericalKMeans(n_clusters = my_n_cluster).fit(X)

from sklearn.cluster import KMeans
obj_kmeans = KMeans(n_clusters = my_n_cluster).fit(X)

hard_labels_skmeans = obj_skmeans.predict(X)
hard_labels_kmeans = obj_kmeans.predict(X)

from sklearn.metrics.cluster import adjusted_rand_score, adjusted_mutual_info_score
ari = adjusted_rand_score(hard_labels_kmeans, hard_labels_skmeans)
ami = adjusted_mutual_info_score(hard_labels_kmeans, hard_labels_skmeans)
print(f'ari: {ari}, ami: {ami}')
