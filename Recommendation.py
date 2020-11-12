from sklearn.neighbors import DistanceMetric, NearestNeighbors
class Recommend:
    search_on = ['age', 'education_field', 'job_role', 'marital_status', 'department',
                'num_companies_worked', 'training_times_last_year',
                'years_at_company', 'years_in_current_role',
                'years_since_last_promotion','years_with_curr_manager']

    analyze_on = ['business_travel', 'daily_rate', 'distance_from_home',
                  'environment_satisfaction', 'hourly_rate', 'job_involvement',
                  'monthly_income', 'monthly_rate', 'over_time', 'percent_salary_hike',
                  'performance_rating', 'relationship_satisfaction', 'stock_option_level',
                  'work_life_balance']
                  
    def __init__(self, data, target):
        self.main_data = data.loc[data[target]==0]
        self.data = data.loc[data[target]==0][self.search_on]
        self.knn = NearestNeighbors().fit(self.data)
    def recommend(self, sample):
        distance, indx = self.knn.kneighbors(sample[self.search_on])
        similar_persons = self.main_data.iloc[indx[0]]
        return similar_persons
