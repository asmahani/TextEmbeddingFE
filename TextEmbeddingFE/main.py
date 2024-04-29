import numpy as np
from sklearn.cluster import KMeans

def embed_text(
    openai_client
    , openai_embedding_model
    , text_list
):
    ret = openai_client.embeddings.create(
        input = text_list
        , model = openai_embedding_model
    )
    return np.transpose(np.array([ret.data[n].embedding for n in range(len(ret.data))]))

def cluster_embeddings(
    X
    , n_clusters
    , n_init
):
    this_kmeans = KMeans(n_clusters = n_clusters, n_init = n_init).fit(X)
    this_labels = this_kmeans.labels_
    return this_labels

def generate_prompt(
    text_list
    , cluster_labels
    , prompt_observations
    , prompt_texts
    , preamble = ''
):
    n_obs = len(text_list)

    if len(cluster_labels) != n_obs:
        raise ValueError("Number of text strings and cluster labels must be the same.")

    my_clusters = np.unique(cluster_labels)
    n_cluster = len(my_clusters)
    check_cluster_labels = np.array_equal(
        my_clusters
        , np.arange(0, n_cluster + 0)
    )
    if not check_cluster_labels:
        raise ValueError("Cluster labels must be integers 0-N, with N >= 1.")
    
    if preamble == '':
        preamble = (f"The following is a list of {str(n_obs)} {prompt_observations}. Text lines represent {prompt_texts}."
                  f" {prompt_observations.capitalize()} have been grouped into {str(n_cluster)} groups, according to their {prompt_texts}."
                  " Please suggest group labels that are representative of their members, and also distinct from each other:"
                 )

    my_body_list = []
    for n in range(n_cluster):
        sublist = [text_list[i] for i in range(n_obs) if cluster_labels[i] == n]
        substring = '\n'.join(sublist)
        substring = 'Group ' + str(n + 1) + ':\n\n' + substring
        my_body_list.append(substring)

    my_body_string = '\n\n=====\n\n'.join(my_body_list)

    my_full_prompt = preamble + '\n\n=====\n\n' + my_body_string

    return my_full_prompt