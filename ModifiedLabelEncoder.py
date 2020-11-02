from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import bisect

class ModifiedLabelEncoder(LabelEncoder):

    def fit_transform(self, X,y=None):
        return self.fit(X, y).transform(X)
    
    def fit(self, X, y=None):
        res = X.copy()
        self.categorical_columns = res.select_dtypes('object').columns
        self.encoders = dict()
        for column in self.categorical_columns:
            le = LabelEncoder()
            le.fit(res[column])
            self.encoders[column]  = le
        return self
    
    def transform(self,X, y=None):
        res = X.copy()
        categorical_columns = res.select_dtypes('object').columns
        for column in categorical_columns:
            res.loc[:,column] = self.encoders[column].transform(res[column]).reshape(-1, 1)
        return res

    def inverse_transform(self,X, y=None):
        res = X.copy()
        for column in res.columns:
            if column in self.categorical_columns:
                res.loc[:, column] = self.encoders[column].inverse_transform(res[column]).reshape(-1, 1)

        return res 

