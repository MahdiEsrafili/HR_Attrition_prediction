
user = 'ilhlwtaaiqfkgy'
pswd = '651864ace94b45772563fe9113fd1eb82062bb401e87b1de7f0fadd585d565d6'
port = 5432
database = 'ddsp5d33leitdc'
host = 'ec2-52-17-53-249.eu-west-1.compute.amazonaws.com'
db_string = f"postgres://{user}:{pswd}@{host}:{port}/{database}"
train_table_name = 'training_data'
predict_table_name = 'prediction_data'
upload_dir = 'upload_dir'
model_dir = 'model_dir/ml_core_model.joblib'
le_dir = 'model_dir/ml_label_encoder.joblib' 