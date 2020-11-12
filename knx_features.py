# train_features = ['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department',
#        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
#        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
#        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
#        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
#        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
#        'RelationshipSatisfaction', 'StockOptionLevel',
#        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
#        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
#        'YearsWithCurrManager']

# predict_features = ['Age', 'BusinessTravel', 'DailyRate', 'Department',
#        'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount',
#        'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
#        'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
#        'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
#        'OverTime', 'PercentSalaryHike', 'PerformanceRating',
#        'RelationshipSatisfaction', 'StockOptionLevel',
#        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
#        'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
#        'YearsWithCurrManager']

# New features:

train_features = [
       'age', 'business_travel', 'daily_rate', 'department',
       'distance_from_home', 'education', 'education_field',
       'environment_satisfaction', 'gender', 'hourly_rate',
       'job_involvement', 'job_level', 'job_role', 'job_satisfaction',
       'marital_status', 'monthly_income', 'monthly_rate',
       'num_companies_worked', 'over18', 'over_time', 'percent_salary_hike',
       'performance_rating', 'relationship_satisfaction', 'standard_hours',
       'stock_option_level', 'total_working_years', 'training_times_last_year',
       'work_life_balance', 'years_at_company', 'years_in_current_role',
       'years_since_last_promotion', 'years_with_curr_manager', 'attrition']


predict_features = [
       'age', 'business_travel', 'daily_rate', 'department',
       'distance_from_home', 'education', 'education_field',
       'environment_satisfaction', 'gender', 'hourly_rate',
       'job_involvement', 'job_level', 'job_role', 'job_satisfaction',
       'marital_status', 'monthly_income', 'monthly_rate',
       'num_companies_worked', 'over18', 'over_time', 'percent_salary_hike',
       'performance_rating', 'relationship_satisfaction', 'standard_hours',
       'stock_option_level', 'total_working_years', 'training_times_last_year',
       'work_life_balance', 'years_at_company', 'years_in_current_role',
       'years_since_last_promotion', 'years_with_curr_manager']