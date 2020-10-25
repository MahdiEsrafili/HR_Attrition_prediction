import pandas as pd

class Analyze:
    def __init__(self, person, recommends):
        self.recommends = recommends
        self.person = person

    def analyze(self):
        numeric_mean = self.recommends.select_dtypes('number').mean().astype('int') 
        numeric_diff = numeric_mean > self.person.select_dtypes('number')
        numeric_diff = numeric_mean[numeric_diff[numeric_diff].dropna(1).columns].to_frame().T
        categorical_frequent = self.recommends.select_dtypes('object').mode() 
        categorical_diff = categorical_frequent != self.person.select_dtypes('object')
        categorical_diff = categorical_frequent[categorical_diff[categorical_diff==True].dropna(1).columns]
        all_diff = pd.concat((numeric_diff, categorical_diff), 1)
        return all_diff

