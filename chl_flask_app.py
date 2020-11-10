from flask import Flask, request, jsonify, redirect, url_for, send_from_directory
import time
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ml_core import ML_core
from Recommendation import Recommend
from recommend_analyzer import Analyze
from sqlalchemy import create_engine 
import sqlalchemy
import joblib
from knx_features import  train_features, predict_features
import chl_app_config
from datetime import datetime

db_string = chl_app_config.db_string
train_table_name = chl_app_config.train_table_name
predict_table_name = chl_app_config.predict_table_name
db = create_engine(db_string) 
metadata = sqlalchemy.MetaData()
metadata.reflect(bind = db)
prediction_table = metadata.tables[predict_table_name]
training_table = metadata.tables[train_table_name]

app = Flask(__name__)

def train(train_data):
    start_time = time.time()
    train_data = train_data[train_features]
    ml_core = ML_core(training_data = train_data)
    training_score = ml_core.train()
    end_time = time.time()
    training_time = end_time - start_time
    joblib.dump(ml_core.model,chl_app_config.model_dir)
    joblib.dump(ml_core.label_encoder, chl_app_config.le_dir )
    return training_time, training_score

def predict(predict_data, train_data):
    start_time = time.time()
    predict_data.set_index('index',inplace = True)
    predict_data = predict_data[predict_features]
    ml_core = ML_core(prediction_data = predict_data)
    ml_core.model = joblib.load(chl_app_config.model_dir)
    ml_core.label_encoder = joblib.load(chl_app_config.le_dir)
    result = ml_core.predict()
    end_time = time.time()
    prediction_time = end_time - start_time
    recommend_time, result = recommend(ml_core, train_data, result)

    #update db
    start_time = time.time()
    result.reset_index(inplace = True)
    for i in range(result.shape[0]):
        indx =result.iloc[i]['index']
        attrition = result.iloc[i]['Attrition']
        rec = result.iloc[i]['recommendations']
        try:
            q = prediction_table.update().where(prediction_table.c.index == str(indx)).values({
            'Attrition' : str(attrition),
            'recommendation': rec,
            'is_sent_to_ml': 'TRUE',
            'updated_by': 'machine learning',
            'updated_time' : str(datetime.now())
            })
            db.execute(q)
        except:
            pass

    end_time = time.time()
    db_update_time = end_time - start_time
    return prediction_time, recommend_time, db_update_time

def recommend(ml_core, train_data, predict_data):
    start_time = time.time()
    need_recommendation = predict_data[predict_data.Attrition==1]
    train_data.set_index('index', inplace = True)
    train_data = train_data[train_features]
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
    return recommend_time, predict_data

@app.route('/')
def home():
    return jsonify(message = 'ml_ai')


@app.route('/train')
def train_requst():
    data_psql = pd.read_sql(f"select * from {train_table_name} where is_sent_to_ml='FALSE'", db)
    training_time, training_score = train(data_psql)
    return jsonify(training_time = '%.2f' % training_time, training_score = '%.2f' % training_score)

@app.route('/predict')
def predict_reques():
    predict_data = pd.read_sql(f"select * from {predict_table_name} where is_sent_to_ml='FALSE'", db)
    train_data = pd.read_sql(f"select * from {train_table_name} where is_sent_to_ml='FALSE'", db)
    prediction_time, recommend_time, db_update_time = predict(predict_data,train_data )
    prediction_time += recommend_time + db_update_time
    return jsonify(prediction_time = '%.2f' % prediction_time)

if __name__ == '__main__':
    app.run(debug=True)