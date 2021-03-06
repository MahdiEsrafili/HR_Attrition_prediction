import pandas as pd

class Analyze:
    analyze_on = ['BusinessTravel', 'DailyRate', 'DistanceFromHome',
                  'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                  'MonthlyIncome', 'MonthlyRate', 'OverTime', 'PercentSalaryHike',
                  'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                  'WorkLifeBalance']
    def __init__(self, person, recommends):
        self.features = list(set(self.analyze_on) & set(recommends.columns) & set(person.columns))
        self.recommends = recommends[self.features]
        self.person = person[self.features]

    def analyze(self):
        numeric_mean = self.recommends.select_dtypes('number').mean().astype('int').to_frame().T
        numeric_diff = numeric_mean > self.person.select_dtypes('number').reset_index(drop = True)
        self.numeric_diff = numeric_mean[numeric_diff[numeric_diff].dropna(1).columns]
        categorical_frequent = self.recommends.select_dtypes('object').mode().reset_index(drop = True).iloc[:1]
        categorical_diff = categorical_frequent != self.person.select_dtypes('object').reset_index(drop = True)
        self.categorical_diff = categorical_frequent[categorical_diff[categorical_diff==True].dropna(1).columns]
        all_diff = pd.concat((self.numeric_diff, self.categorical_diff), 1)
        return all_diff.iloc[0].to_json()

