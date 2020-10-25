from sklearn.neighbors import DistanceMetric, NearestNeighbors
class Recommend:
    search_on = ['Age', 'EducationField', 'MaritalStatus', 'Department',
                'NumCompaniesWorked', 'TrainingTimesLastYear',
                'YearsAtCompany', 'YearsInCurrentRole',
                'YearsSinceLastPromotion','YearsWithCurrManager']
    def __init__(self, data, target):
        self.main_data = data.loc[data[target]==0]
        self.data = data.loc[data[target]==0][self.search_on]
        self.knn = NearestNeighbors().fit(self.data)
    def recommend(self, sample):
        distance, indx = self.knn.kneighbors(sample[self.search_on])
        return self.main_data.iloc[indx[0]]