import numpy as np
from sklearn.cluster import KMeans
import tiktoken
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import copy
import openai
import pandas as pd
from scipy.stats import fisher_exact
from sentence_transformers import SentenceTransformer

class FeatureExtractor_BinaryClassifier:
    """
    Class for extracting a single-column feature from text embeddings by wrapping a K-Nearest-Neighbor binary classifier
    in K-fold cross-validation. For in-sample data, i.e., the data used to train the KNN model, the feature
    is the prediction of the CV iteration where data point was left out of training. For out-of-sample
    data, the feature is the average predictions of K individual models trained during CV.

    Attributes:
        knn (KNeighborsClassifier): The KNN classifier.
        logit (bool): Boolean flag indicating whether predictions should be returned on a logit scale or not.
        laplace (bool): Boolean flag, indicating whether lplace smoothing should be applied to the predicted probabilities or not.
        kfolds (KFold): The cross-validation fold generator.
        nfolds (int): The number of folds used for cross-validation.
        trained_models (list): List of trained KNN models, one per fold.
        insample_prediction_proba (numpy.ndarray): Array of in-sample prediction probabilities.

    :param kwargs: Keyword arguments passed to the KNeighborsClassifier constructor.
    """
    def __init__(self, logit = True, laplace = True, **kwargs):
        self.knn = KNeighborsClassifier(**kwargs)
        self.logit = logit
        self.laplace = laplace
        return None
    
    def fit(self, X, y, cv = 5):
        """
        Fits the KNN model using K-fold cross-validation, and generate and save the in-sample
        predicted probabilities.

        :param X: Feature matrix.
        :type X: numpy.ndarray
        :param y: Target vector.
        :type y: numpy.ndarray
        :param cv: Number of cross-validation folds.
        :type cv: int
        :return: Self.
        :rtype: FeatureExtractor_BinaryClassifier

        :raises ValueError: If `X` and `y` have mismatched lengths.
        :raises TypeError: If `cv` is not an integer.
        """
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError("X and y must be numpy arrays.")
        if len(X) != len(y):
            raise ValueError("The length of X and y must be the same.")
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
            
            tmp_pred = tmp_knn.predict_proba(X[test_index, :])[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            insample_prediction_proba[test_index] = tmp_pred
            
            trained_models.append(tmp_knn)

        self.trained_models = trained_models
        self.insample_prediction_proba = insample_prediction_proba
        return self
    
    def predict_proba(self, X = None):
        """
        Predicts probabilities using the trained models. If `X` is None, returns in-sample predictions.
        Otherwise, average of predictions from each individual model trained during cross-validated fit
        is returned.

        :param X: Feature matrix; if None, returns in-sample prediction probabilities.
        :type X: numpy.ndarray, optional
        :return: Predicted probabilities.
        :rtype: numpy.ndarray

        :raises RuntimeError: If called before the model is fit.
        """

        if not hasattr(self, 'trained_models'):
            raise RuntimeError("This FeatureExtractor_BinaryClassifier instance is not fitted yet. Call 'fit' with appropriate arguments before using this method.")
        if X is None:
            return self.insample_prediction_proba
        if not isinstance(X, np.ndarray):
            raise TypeError("X must be a numpy array.")       
        
        all_preds = np.empty((X.shape[0], self.nfolds), dtype = float)
        for n in range(self.nfolds):
            tmp_pred = self.trained_models[n].predict_proba(X)[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            all_preds[:, n] = tmp_pred
        return np.mean(all_preds, axis = 1)

def embed_text_wrapper(
    text_list
    , model_repository = 'openai'
    , model_name = 'text-embedding-3-large'
    , openai_client = None
):
    if not isinstance(text_list, list) or not all(isinstance(item, str) for item in text_list):
        raise ValueError("text_list must be a list of strings")

    if not text_list:
        raise ValueError("text_list cannot be empty")
    
    if model_repository == 'openai':
        ret = openai_client.embeddings.create(
            input = text_list
            , model = model_name
        )
        return np.array([ret.data[n].embedding for n in range(len(ret.data))])
    elif model_repository == 'huggingface':
        model = SentenceTransformer(model_name)
        return model.encode(text_list) 
    else:
        raise ValueError("model repository must be one of 'openai' or 'huggingface'.")
    

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

import numpy as np

class SphericalKMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_init = n_init
        self.total_within_cluster_distance_ = None

    def fit(self, X):
        best_total_within_cluster_distance = float('inf')
        best_centroids = None
        best_labels = None

        for _ in range(self.n_init):
            if self.random_state is not None:
                np.random.seed(self.random_state)
            
            # Normalize the data
            X = self._normalize(X)
            
            # Initialize centroids
            self.centroids = self._initialize_centroids(X)
            
            for _ in range(self.max_iter):
                # Assign clusters
                self.labels_ = self._assign_clusters(X)
                # Compute new centroids
                new_centroids = self._compute_centroids(X)
                # Check for convergence
                if np.all(np.abs(new_centroids - self.centroids) <= self.tol):
                    break
                self.centroids = new_centroids

            # Calculate total within-cluster distance
            total_within_cluster_distance = self._calculate_total_within_cluster_distance(X)
            
            if total_within_cluster_distance < best_total_within_cluster_distance:
                best_total_within_cluster_distance = total_within_cluster_distance
                best_centroids = self.centroids
                best_labels = self.labels_

        # Set the best results
        self.centroids = best_centroids
        self.labels_ = best_labels
        self.total_within_cluster_distance_ = best_total_within_cluster_distance

    def predict(self, X):
        X = self._normalize(X)
        return self._assign_clusters(X)

    def _initialize_centroids(self, X):
        # Randomly select n_clusters points as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X):
        # Calculate the cosine similarity between each point and each centroid
        similarities = np.array([[self._calculate_similarity(x, centroid) for centroid in self.centroids] for x in X])
        # Assign each point to the nearest centroid (highest similarity)
        return np.argmax(similarities, axis=1)

    def _compute_centroids(self, X):
        # Compute the mean of the points in each cluster to find the new centroids
        centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            if len(cluster_points) > 0:
                centroids[k] = self._normalize(cluster_points.mean(axis=0).reshape(1, -1))
        return centroids

    def _calculate_similarity(self, x1, x2):
        # Cosine similarity
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def _normalize(self, X):
        # Normalize the rows of X to have unit norm
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    def _calculate_total_within_cluster_distance(self, X):
        total_distance = 0
        for k in range(self.n_clusters):
            cluster_points = X[self.labels_ == k]
            centroid = self.centroids[k]
            distances = [1 - self._calculate_similarity(point, centroid) for point in cluster_points]
            total_distance += np.sum(distances)
        return total_distance

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

def cluster_embeddings(
    X
    , n_clusters
    , n_init
    #, spherical = True
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

    spherical = False # SphericalKMeans class must be debugged first
    
    if not isinstance(X, np.ndarray):
        raise ValueError("X must be a numpy ndarray.")
    
    if not X.ndim == 2:
        raise ValueError("X must be a 2D numpy array.")

    if not isinstance(n_clusters, int) or n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")

    if not isinstance(n_init, int) or n_init <= 0:
        raise ValueError("n_init must be a positive integer.")
    
    try:
        if spherical:
            return SphericalKMeans(n_clusters = n_clusters, n_init = n_init).fit_predict(X)
        else:
            return KMeans(n_clusters = n_clusters, n_init = n_init).fit_predict(X)
        #return this_kmeans.labels_
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
    """
    Calculate the maximum number of tokens for any text in the list using 
    the tokenizer associated with the specified OpenAI embedding model, and 
    the total number of tokens for all texts using the tokenizer associated 
    with the specified text-completion OpenAI model.

    :param text_list: List of text strings to be encoded and counted for tokens.
    :type text_list: list of str
    :param openai_embedding_model: The OpenAI embedding model whose tokenizer is applied to the text in order 
    to find the maximum number of tokens in any single text.
    :type openai_embedding_model: str
    :param openai_textgen_model: The OpenAI text-completion model whose tokenizer is applied to the text 
    in order to find the total number of tokens across all texts.
    :type openai_textgen_model: str
    :return: A tuple containing the maximum number of tokens in any single text and the total number of tokens across all texts.
    :rtype: (int, int)

    :Example:

    >>> text_list = ["hello world", "example of a longer piece of text that needs tokenizing"]
    >>> count_tokens(text_list)
    (11, 13)
    """
    
    # Validate input type for text_list
    if not isinstance(text_list, list) or not all(isinstance(text, str) for text in text_list):
        raise ValueError("text_list must be a list of strings.")
    
    # Ensure text_list is not empty to avoid errors in max() and sum() calculations
    if not text_list:
        raise ValueError("text_list cannot be empty.")
    
    try:
        encoder_embedding = tiktoken.encoding_for_model(openai_embedding_model)
        encoder_textgen = tiktoken.encoding_for_model(openai_textgen_model)
        
        ntokens_embedding_list = [len(encoder_embedding.encode(text)) for text in text_list]
        ntokens_embedding_max = max(ntokens_embedding_list)
        
        ntokens_textgen_list = [len(encoder_textgen.encode(text)) for text in text_list]
        ntokens_textgen_total = sum(ntokens_textgen_list)
        
        return (ntokens_embedding_max, ntokens_textgen_total)
    except Exception as e:
        raise RuntimeError(f"An error occurred while executing the function: {str(e)}")

def interpret_clusters(
    openai_client
    , prompt
    , openai_textgen_model = 'gpt-4-turbo'
    , temperature = 1.0
):
    """
    Thin wrapper around OpenAI's text completion API call, to be used for submitting
    clustered text fields for cluster labeling.
    
    :param openai_client: An instance of the OpenAI API client.
    :type openai_client: OpenAI.Client
    :param prompt: The prompt text to send to the model.
    :type prompt: str
    :param openai_textgen_model: The identifier for the OpenAI text generation model to use.
    :type openai_textgen_model: str, optional
    :param temperature: The temperature setting for the text generation, which controls randomness.
    :type temperature: float, optional
    :return: The text completion generated by the model.
    :rtype: str
    
    :raises ValueError: If the prompt is empty or not a string.
    :raises RuntimeError: If there is an issue with the API call.
    """
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Prompt must be a non-empty string.")
    
    try:
        response = openai_client.chat.completions.create(
            model = openai_textgen_model
            , messages = [
                {"role": "user", "content": prompt}
            ]
            , temperature = temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"Failed to generate completion from OpenAI API: {e}")

def fisher_test_wrapper(
    dat
    , col_x = None
    , col_y = None
):
    if not col_x:
        col_x = dat.columns[0]
    if not col_y:
        col_y = dat.columns[1]
    
    results = []
    categories = dat[col_x].unique()
    
    for category in categories:
        # Create contingency table for each category vs. rest
        table = pd.crosstab(dat[col_x] == category, dat[col_y])
        # Calculate Fisher's Exact Test
        odds_ratio, p_value = fisher_exact(table, alternative='two-sided')
        results.append((category, odds_ratio, p_value))
    
    results_df = pd.DataFrame(results, columns=['Category', 'Odds Ratio', 'P-value'])
    return results_df
