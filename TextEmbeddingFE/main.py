import numpy as np
from sklearn.cluster import KMeans
import tiktoken

def embed_text(
    openai_client
    , text_list
    , openai_embedding_model = 'text-embedding-3-large'
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
    , openai_textgen_model = 'gpt-4-turbo'
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
    
    encoder = tiktoken.encoding_for_model(openai_textgen_model)
    ntokens = len(encoder.encode(my_full_prompt))
    
    return (ntokens, my_full_prompt)

def count_tokens(
    text_list
    , openai_embedding_model = 'text-embedding-3-large'
    , openai_textgen_model = 'gpt-4-turbo'
):
    encoder_embedding = tiktoken.encoding_for_model(openai_embedding_model)
    encoder_textgen = tiktoken.encoding_for_model(openai_textgen_model)
    
    ntokens_embedding_list = [len(encoder_embedding.encode(text)) for text in text_list]
    ntokens_embedding_max = max(ntokens_embedding_list)
    
    ntokens_textgen_list = [len(encoder_textgen.encode(text)) for text in text_list]
    ntokens_textgen_total = sum(ntokens_textgen_list)
    
    return (ntokens_embedding_max, ntokens_textgen_total)

def interpret_clusters(
    openai_client
    , prompt
    , openai_textgen_model = 'gpt-4-turbo'
    , temperature = 1.0
):
    response = openai_client.chat.completions.create(
        model = openai_textgen_model
        , messages = [
            {"role": "user", "content": prompt}
        ]
        , temperature = temperature
    )
    return response.choices[0].message.content

