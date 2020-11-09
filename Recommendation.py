from sklearn.neighbors import DistanceMetric, NearestNeighbors
class Recommend:
    search_on = ['Age', 'EducationField', 'JobRole', 'MaritalStatus', 'Department',
                'NumCompaniesWorked', 'TrainingTimesLastYear',
                'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion','YearsWithCurrManager']

    analyze_on = ['BusinessTravel', 'DailyRate', 'DistanceFromHome',
                  'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement',
                  'MonthlyIncome', 'MonthlyRate', 'OverTime', 'PercentSalaryHike',
                  'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                  'WorkLifeBalance']
    def __init__(self, data, target):
        self.main_data = data.loc[data[target]==0]
        self.data = data.loc[data[target]==0][self.search_on]
        self.knn = NearestNeighbors().fit(self.data)
    def recommend(self, sample):
        distance, indx = self.knn.kneighbors(sample[self.search_on])
        similar_persons = self.main_data.iloc[indx[0]]
        return similar_persons


