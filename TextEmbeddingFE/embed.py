import pandas as pd
import numpy as np
import openai
from sklearn.base import BaseEstimator, TransformerMixin
from sentence_transformers import SentenceTransformer
from gensim.utils import simple_preprocess
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
import vertexai

class Textcol2Mat(BaseEstimator, TransformerMixin):
    def __init__(
        self
        , type = 'doc2vec'
        , openai_client = None
        , google_project_id = None
        , google_location = 'us-central1'
        , embedding_model_openai = 'text-embedding-3-large'
        , embedding_model_st = 'neuml/pubmedbert-base-embeddings-matryoshka'
        , embedding_model_doc2vec = 'PV-DM'
        , embedding_model_google = 'text-embedding-004'
        , doc2vec_epochs = 40
        , doc2vec_vector_size = 10
        , google_task = 'SEMANTIC_SIMILARITY'
        , google_batch_size = 250
        #, google_vector_size = 256
        , colsep = ' || '
        , return_cols_prefix = 'X_'
    ):
        if not (type in ('openai', 'st', 'doc2vec', 'google')):
            raise ValueError('Invalid embedding type')
        if not (embedding_model_doc2vec in ('PV-DM', 'PV-DBOW')):
            raise ValueError('Doc2Vec model must be one of "PV-DM" or "PV-DBOW"')
        
        self.type = type
        self.openai_client = openai_client
        self.google_project_id = google_project_id
        self.google_location = google_location
        self.embedding_model_openai = embedding_model_openai
        self.embedding_model_st = embedding_model_st
        self.embedding_model_doc2vec = embedding_model_doc2vec
        self.embedding_model_google = embedding_model_google
        self.colsep = colsep
        self.return_cols_prefix = return_cols_prefix
        self.doc2vec_epochs = doc2vec_epochs
        self.doc2vec_vector_size = doc2vec_vector_size
        self.google_task = google_task
        self.google_batch_size = google_batch_size
        #self.google_vector_size = google_vector_size
        
        pass
    
    def _fit_doc2vec(self, X, y = None):
        if not (X.dtypes == 'object').all():
            raise TypeError('All columns of X must be of string (object) type')
        
        Xstr = X.fillna('').astype(str).apply(
            lambda row: self.colsep.join([f'{col}: {row[col]}' for col in X.columns])
            , axis=1
        ).tolist()
        
        corpus = [TaggedDocument(words = simple_preprocess(doc), tags=[str(i)]) for i, doc in enumerate(Xstr)]
        alg = 1 if self.embedding_model_doc2vec == 'PV-DM' else 0
        model = Doc2Vec(corpus, vector_size = self.doc2vec_vector_size, window=2, min_count=1, epochs = self.doc2vec_epochs, dm = alg)
        self.doc2vec_model = model
        return self
    
    def fit(self, X, y = None):
        if self.type == 'doc2vec':
            return self._fit_doc2vec(X, y)
        else:
            return self

    def fit_transform(self, X, y = None):
        if self.type == 'doc2vec':
            self._fit_doc2vec(X, y)
        return self.transform(X)
    
    def transform(self, X):
        
        if not (X.dtypes == 'object').all():
            raise TypeError('All columns of X must be of string (object) type')
        
        Xstr = X.fillna('').astype(str).apply(
            lambda row: self.colsep.join([f'{col}: {row[col]}' for col in X.columns])
            , axis=1
        ).tolist()
        
        if self.type == 'openai':
            arr = self._transform_openai(Xstr)
        elif self.type == 'st':
            arr = self._transform_st(Xstr)
        elif self.type == 'doc2vec':
            arr = self._transform_doc2vec(Xstr)
        elif self.type == 'google':
            arr = self._transform_google(Xstr)
        else:
            raise ValueError('Invalid embedding type')
        return pd.DataFrame(arr, columns = [self.return_cols_prefix + str(i) for i in range(arr.shape[1])])

    def _transform_google(self, X):
        vertexai.init(project=self.google_project_id, location=self.google_location)
        model = TextEmbeddingModel.from_pretrained(self.embedding_model_google)
        if self.embedding_model_google == 'textembedding-gecko@001':
            inputs = [TextEmbeddingInput(text) for text in X]
        else:
            inputs = [TextEmbeddingInput(text, self.google_task) for text in X]
        kwargs = {}
        embeddings = []
        batch_size = self.google_batch_size
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            batch_embeddings = model.get_embeddings(batch_inputs, **kwargs)
            embeddings.extend(batch_embeddings)
        return np.array([embedding.values for embedding in embeddings])
    
    def _transform_doc2vec(self, X):
        out = [self.doc2vec_model.infer_vector(simple_preprocess(doc)) for doc in X]
        return np.array(out)

    def _transform_st(self, X):
        model_name = self.embedding_model_st
        model_name_split = model_name.split('@')
        assert len(model_name_split) <= 2, 'Too many @ characters in model name'
        if len(model_name_split) == 1:
            model = SentenceTransformer(model_name_split[0])
        else:
            model = SentenceTransformer(model_name_split[0], revision = model_name_split[1])
        return model.encode(X) 
    
    def _transform_openai(self, X):
        if not self.openai_client:
            raise ValueError('Invalid OpenAI client')
        
        ret = self.openai_client.embeddings.create(
            input = X
            , model = self.embedding_model_openai
        )
        ret = np.array([ret.data[n].embedding for n in range(len(ret.data))])
        return ret
