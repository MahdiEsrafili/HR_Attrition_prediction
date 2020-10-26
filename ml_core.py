import joblib
import numpy as np
import pandas as pd 
from ModifiedLabelEncoder import ModifiedLabelEncoder
from Recommendation import Recommend
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline

class ML_core:
    def __init__(self, traing_data, prediction_data):
        self.training_data = traing_data
        self.prediction_data = prediction_data

    def train(self):
        self.label_encoder = Pipeline(steps=[
            ('label_encoding',ModifiedLabelEncoder())])
        data = self.pure_data(self.training_data)
        pure_data_led= self.label_encoder.fit_transform(data)
        self.model = Pipeline(steps=[
            ('model_xgb', XGBClassifier(random_state=42))
            ])
        self.model = XGBClassifier()
        self.model.fit(pure_data_led.drop('Attrition', 1),pure_data_led.Attrition)
        # self.pipeline = union pipelines
        # self.save_model(self.pipeline)

    

    def predict(self):
        data = self.pure_data(self.prediction_data)
        pure_data_led = self.label_encoder.transform(data)
        result = self.model.predict(pure_data_led)
        self.prediction_data['Attrition'] = result
        return self.prediction_data
        ## reverse encode the attrition result
        ## send result to kafka connector

    def pure_data(self, data):
        self.extra_columns = ['EmployeeCount', 'EmployeeNumber']
        return data.drop(self.extra_columns, 1)

    def save_model(self):
        pass

    def load_model(self):
        pass

