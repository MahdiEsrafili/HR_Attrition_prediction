import joblib
import numpy as np
import pandas as pd 
from ModifiedLabelEncoder import ModifiedLabelEncoder
from Recommendation import Recommend
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

class ML_core:
    def __init__(self, training_data= None, prediction_data=None):
        self.training_data = training_data
        self.prediction_data = prediction_data

    def train(self):
        self.label_encoder = Pipeline(steps=[
            ('label_encoding',ModifiedLabelEncoder())])
        # if self.training_data != None:
        data = self.pure_data(self.training_data)
        pure_data_led= self.label_encoder.fit_transform(data)
        # self.model = Pipeline(steps=[
        #     ('model_xgb', XGBClassifier(min_child_weight=25, n_estimators = 3000,
        #                 colsample_bynode= 0.6, max_depth = 8,
        #                 random_state= 42))
        #     ])
        self.model = XGBClassifier(min_child_weight=25, n_estimators = 3000,
                        colsample_bynode= 0.6, max_depth = 8,
                        random_state= 42)
        self.model.fit(pure_data_led.drop('Attrition', 1),pure_data_led.Attrition)
        return self.model.score(pure_data_led.drop('Attrition', 1),pure_data_led.Attrition)
        # else:
        #     return 'nothin_to_train'
            # self.pipeline = union pipelines
            # self.save_model(self.pipeline)

    

    def predict(self):
        # if self.prediction_data:
        data = self.pure_data(self.prediction_data)
        pure_data_led = self.label_encoder.transform(data)
        result = self.model.predict(pure_data_led)
        self.prediction_data['Attrition'] = result
        return self.prediction_data
        # else:
        #     return 'nothing_to_predict'
        ## reverse encode the attrition result
        ## send result to kafka connector

    def pure_data(self, data):
        # self.extra_columns = ['EmployeeCount', 'EmployeeNumber']
        # return data.drop(self.extra_columns, 1)
        return data

    def save_model(self):
        pass

    def load_model(self):
        pass

