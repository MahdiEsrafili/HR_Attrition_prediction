from flask import Flask, request, jsonify, redirect, url_for, send_from_directory
import time
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ml_core import ML_core
from Recommendation import Recommend
from recommend_analyzer import Analyze
from sqlalchemy import create_engine 
import joblib

db_string = "postgres://mluser:123456789@localhost:5432/mldb"
train_table_name = 'training_data'
predict_table_name = 'prediction_data'
db = create_engine(db_string)  

upload_dir = 'upload_dir'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_dir

def train(train_data):
    start_time = time.time()
    none_informative = ['Name', 'index']
    train_data = train_data.drop(none_informative, 1)
    ml_core = ML_core(training_data = train_data)
    training_score = ml_core.train()
    end_time = time.time()
    training_time = end_time - start_time
    joblib.dump(ml_core.model,'model_dir/ml_core_model.joblib')
    joblib.dump(ml_core.label_encoder, 'model_dir/ml_label_encoder.joblib' )
    return training_time, training_score

def predict(predict_data, train_data):
    start_time = time.time()
    none_informative = ['Name', 'index']
    predict_data = predict_data.drop(none_informative, 1)
    ml_core = ML_core(prediction_data = predict_data)
    ml_core.model = joblib.load('model_dir/ml_core_model.joblib')
    ml_core.label_encoder = joblib.load('model_dir/ml_label_encoder.joblib')
    result = ml_core.predict()
    end_time = time.time()
    prediction_time = end_time - start_time
    recommend_time = recommend(ml_core, train_data, result)
    return prediction_time, recommend_time

def recommend(ml_core, train_data, predict_data):
    start_time = time.time()
    need_recommendation = predict_data[predict_data.Attrition==1]
    none_informative = ['Name', 'index']
    train_data = train_data.drop(none_informative, 1)
    train_data_led = ml_core.label_encoder.transform(train_data)
    recommender = Recommend(train_data_led, 'Attrition')
    recommends_list = []
    for person_indx in range(need_recommendation.shape[0]):
        person = ml_core.label_encoder.transform(need_recommendation.iloc[person_indx:person_indx+1])
        recoms = recommender.recommend(person)
        analyzer = Analyze(need_recommendation.iloc[person_indx:person_indx+1], train_data.iloc[recoms.index])
        an = analyzer.analyze()
        recommends_list.append(an)

    predict_data['recommendations'] = 'NaN'
    predict_data.loc[predict_data.Attrition==1, 'recommendations'] = recommends_list
    end_time = time.time()
    recommend_time = end_time - start_time
    return recommend_time

@app.route('/')
def home():
    return jsonify(message = 'ml_ai')


@app.route('/train')
def train_requst():
    data_psql = pd.read_sql(f"select * from {train_table_name} where is_sent_to_ml='FALSE'", db)
    training_time, training_score = train(data_psql)
    return jsonify(row_count = data_psql.shape, training_time = '%.2f' % training_time, training_score = '%.2f' % training_score)

@app.route('/predict')
def predict_reques():
    predict_data = pd.read_sql(f"select * from {predict_table_name} where is_sent_to_ml='FALSE'", db)
    train_data = pd.read_sql(f"select * from {train_table_name} where is_sent_to_ml='FALSE'", db)
    prediction_time, recommend_time = predict(predict_data,train_data )
    return jsonify(prediction_time = '%.2f' % prediction_time, recommend_time = '%.2f' % recommend_time)

if __name__ == '__main__':
    app.run(debug=True)