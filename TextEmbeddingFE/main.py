import numpy as np
from sklearn.cluster import KMeans
import tiktoken
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import copy
import openai

def embed_text(
    openai_client
    , text_list
    , openai_embedding_model = 'text-embedding-3-large'
):
    """Thin wrapper around OpenAI's API call for embedding text.

    :param openai_client: Active OpenAI client connection
    :param text_list: List of strings to be embedded
    :type text_list: list
    :param openai_embedding_model: Name of OpenAI embedding model, defaults to 'text-embedding-3-large'
    :type openai_embedding_model: str
    :return: Embedding matrix, a 2D numpy array, with each row being the embedded vector corresponding to an element of text_list
    :rtype: numpy.ndarray
    """
    if not isinstance(text_list, list) or not all(isinstance(item, str) for item in text_list):
        raise ValueError("text_list must be a list of strings")

    if not text_list:
        raise ValueError("text_list cannot be empty")
    
    try:
        ret = openai_client.embeddings.create(
            input = text_list
            , model = openai_embedding_model
        )
        return np.array([ret.data[n].embedding for n in range(len(ret.data))])
    except openai.OpenAIError as e:
        raise RuntimeError(f"An error occurred while processing the request: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {str(e)}") from e
        

def cluster_embeddings(
    X
    , n_clusters
    , n_init
):
    """
    Thin wrapper around the KMeans clustering algorithm in sklearn, returning cluster labels.

    :param X: The embedding matrix, a 2D numpy array, such as one returned by :func:`embed_text`, 
    where each row represents the embedding vector for one observation.
    :type X: numpy.ndarray
    :param n_clusters: Number of clusters to create.
    :type n_clusters: int
    :param n_init: Number of runs of k-means clustering to perform, each with a different random 
    initialization of cluster centers.
    :type n_init: int
    :return: Vector of cluster labels, one for each row of X
    :rtype: numpy.ndarray
    """
    this_kmeans = KMeans(n_clusters = n_clusters, n_init = n_init).fit(X)
    this_labels = this_kmeans.labels_
    return this_labels

def generate_prompt(
    text_list
    , cluster_labels
    , prompt_observations = None
    , prompt_texts = None
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

class FeatureExtractor_Classifier:
    def __init__(self, **kwargs):
        self.knn = KNeighborsClassifier(**kwargs)
        return None
    
    def fit(self, X, y, cv = 5):

        if not isinstance(cv, int):
            raise TypeError("'cv' must be an integer")
        
        # create folds
        kf = KFold(n_splits = cv)
        kf.get_n_splits(X)
        self.kfolds = kf
        self.nfolds = cv
        
        # train model within each fold
        trained_models = []
        insample_prediction_proba = np.empty(len(y), dtype = float)
        for (train_index, test_index) in kf.split(X):
            tmp_knn = copy.deepcopy(self.knn).fit(X[train_index, :], y[train_index])
            insample_prediction_proba[test_index] = tmp_knn.predict_proba(X[test_index, :])[:, 1]
            trained_models.append(tmp_knn)

        self.trained_models = trained_models
        self.insample_prediction_proba = insample_prediction_proba
        return self
    
    def predict_proba(self, X = None):

        if X is None:
            return self.insample_prediction_proba
        
        all_preds = np.empty((X.shape[0], self.nfolds), dtype = float)
        for n in range(self.nfolds):
            all_preds[:, n] = self.trained_models[n].predict_proba(X)[:, 1]
        return np.mean(all_preds, axis = 1)
