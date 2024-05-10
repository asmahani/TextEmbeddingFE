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
    :raises ValueError: If `text_list` is not a list of strings or is empty.
    :raises RuntimeError: If OpenAI API returns with an error, or any other unexpected errors occur.
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
    :raises ValueError: If `X` is not a 2D numpy array, or if `n_clusters` or `n_init` are not positive integers.
    :raises RuntimeError: If an unexpected error occurs during the KMeans clustering process.
    """
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy ndarray.")
    
    if not X.ndim == 2:
        raise ValueError("X must be a 2D numpy array.")

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    if not isinstance(n_init, int) or n_init <= 0:
        raise ValueError("n_init must be a positive integer.")
    
    try:
        this_kmeans = KMeans(n_clusters = n_clusters, n_init = n_init).fit(X)
        return this_kmeans.labels_
    except Exception as e:
        raise RuntimeError(f"An error occurred while performing KMeans clustering: {e}")

def generate_prompt(
    text_list
    , cluster_labels
    , prompt_observations = None
    , prompt_texts = None
    , preamble = ''
    , openai_textgen_model = 'gpt-4-turbo'
):
    """Assembling a prompt to solicit cluster descriptions from OpenAI's text completion models.
    The returned prompt will consist of two parts: 1- preamble, which provides the context and instructions to
    the text-completion model, and 2- the list of observations, grouped by their clusters, and
    each observation represented by the value of their text field.

    :param text_list: List of text strings associated with a collection of observations.
    :type text_list: list
    :param cluster_labels: Numpy array of cluster memberships associated the same collection of observations.
    :type cluster_labels: numpy.ndarray
    :param prompt_observations: What observation units should be referred to in the prompt, must be in plural form.
    Used to generate the prompt preamble, and will be ignored if `preamble` is any string other than the empty string.
    :type prompt_observations: str
    :param prompt_texts: What does the text field represent for each observation unit? Must be in plural form.
    Used to generate the prompt preamble, and will be ignored if `preamble` is any string other than the empty string.
    :type prompt_texts: str
    :param preamble: Prompt preamble which provides the context and requested task to the text-completion model.
    If an empty string is provided - which is the default value - preamble will be automatically constructed.
    :type preamble: str
    :param openai_textgen_model: Name of target text-completion model from OpenAI, defaults to 'gpt-4-turbo'
    :type openai_textgen_model: str
    :return: Tuple consisting of 1- length of the assembled prompt, 2- full text of the prompt.
    :rtype: (int, str)
    """
    if not isinstance(text_list, list) or not all(isinstance(item, str) for item in text_list):
        raise TypeError("text_list must be a list of strings.")
    if not isinstance(cluster_labels, np.ndarray):
        raise TypeError("cluster_labels must be a numpy ndarray.")
    if not text_list:
        raise ValueError("text_list cannot be empty.")
    if cluster_labels.size == 0:
        raise ValueError("cluster_labels cannot be empty.")
    n_obs = len(text_list)
    if len(cluster_labels) != n_obs:
        raise ValueError("Number of text strings and cluster labels must be the same.")
    if preamble == '' and (not prompt_observations or not prompt_texts):
        raise ValueError("'prompt_observations' and 'prompt_texts' must be provided if 'preamble' is to be generated automatically.")

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
