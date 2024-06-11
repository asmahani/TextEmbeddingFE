from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils.multiclass import type_of_target
import numpy as np
from sklearn.model_selection import KFold
import copy

class Mat2ScoreBase(BaseEstimator, TransformerMixin):
    def __init__(self, nx = None, ncv = 5):
        self.nx = nx
        self.ncv = ncv
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X
        
class Mat2ScoreClassifier(Mat2ScoreBase, ClassifierMixin):
    def __init__(self, base_learner = KNeighborsClassifier(), nx = None, ncv = 5, logit = True, laplace = True, **kwargs):
        super().__init__(nx = nx, ncv = ncv)
        self.base_learner = base_learner
        self.knn = KNeighborsClassifier(**kwargs)
        self.logit = logit
        self.laplace = laplace
        pass

    def fit(self, X, y):
        if not self.nx:
            self.nx = X.shape[1]
        
        if self.nx > X.shape[1]:
            raise ValueError('X has fewer columns than nx')
        
        X, y = check_X_y(X, y)
        if type_of_target(y) != 'binary':
            raise ValueError('Target type must be binary')
        
        # select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])
        
        # create folds
        kf = KFold(n_splits = self.ncv, shuffle = True)
        kf.get_n_splits(X)
        self.kfolds = kf
        
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
        self.insample_prediction_proba = np.reshape(insample_prediction_proba, (insample_prediction_proba.size, 1))
        return self

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.insample_prediction_proba
        
    def transform(self, X):
        X = check_array(X)
        
        # select subset of columns and renormalize
        X = np.apply_along_axis(lambda x: x / np.sqrt(np.sum(x * x)), 1, X[:, :self.nx])
        
        all_preds = np.empty((len(X), self.ncv), dtype = float)
        for n in range(self.ncv):
            tmp_pred = self.trained_models[n].predict_proba(X)[:, 1]
            if self.laplace:
                tmp_pred = (tmp_pred * self.knn.n_neighbors + 1) / (self.knn.n_neighbors + 2)
            if self.logit:
                tmp_pred = np.log(tmp_pred / (1.0 - tmp_pred))
            all_preds[:, n] = tmp_pred
        ret = np.mean(all_preds, axis = 1)
        return np.reshape(ret, (ret.size, 1))

    def predict(self, X):
        ret = self.predict_proba(X)
        return np.where(ret < 0.5, 0, 1)
    
    def predict_proba(self, X):
        ret = self.transform(X)
        if self.logit:
            ret = 1.0 / (1.0 + np.exp(-ret))
        return ret


