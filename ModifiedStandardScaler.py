from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import bisect

class ModifiedStandardScaler(StandardScaler):

    def fit_transform(self, X,y=None):
        return self.fit(X, y).transform(X)
    
    def fit(self, X, y=None):
        res = X.copy()
        numeric_columns = res.select_dtypes('number').columns
        self.encoders = dict()
        for column in numeric_columns:
            le = StandardScaler()
            le.fit(res[column].values.reshape((-1, 1)))
            self.encoders[column]  = le
        return self
    
    def transform(self,X, y=None):
        res = X.copy()
        numeric_columns = res.select_dtypes('number').columns
        for column in numeric_columns:
            res.loc[:,column] = self.encoders[column].transform(res[column].values.reshape((-1, 1))).reshape(-1, 1)
        return res