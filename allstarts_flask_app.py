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
import stars_ap_config as appconfig
from datetime import datetime
import os

db_string = appconfig.db_string
train_table_name = appconfig.train_table_name
predict_table_name = appconfig.predict_table_name
db = create_engine(db_string) 
conn = db.connect()
metadata = sqlalchemy.MetaData()
metadata.reflect(bind = conn)

prediction_table = metadata.tables[predict_table_name] 
training_table = metadata.tables[train_table_name]

if not os.path.exists('model_dir'):
    os.makedirs('model_dir')

app = Flask(__name__)

def train(train_data):
    start_time = time.time()
    train_data = train_data[train_features]
    ml_core = ML_core(training_data = train_data)
    training_score = ml_core.train()
    end_time = time.time()
    training_time = end_time - start_time
    joblib.dump(ml_core.model,appconfig.model_dir)
    joblib.dump(ml_core.label_encoder, appconfig.le_dir )
    return training_time, training_score

def predict(predict_data, train_data):
    start_time = time.time()
    predict_data.set_index('id',inplace = True)
    predict_data = predict_data[predict_features]
    ml_core = ML_core(prediction_data = predict_data)
    try:
        ml_core.model = joblib.load(appconfig.model_dir)
        ml_core.label_encoder = joblib.load(appconfig.le_dir)
    except:
        message = 'FAILURE'
        rediction_time = 0
        recommend_time = 0
        db_update_time = 0
        return rediction_time, recommend_time, db_update_time, message
    result = ml_core.predict()
    end_time = time.time()
    prediction_time = end_time - start_time
    recommend_time, result = recommend(ml_core, train_data, result)

    #update db
    start_time = time.time()
    result.reset_index(inplace = True)
    for i in range(result.shape[0]):
        indx =result.iloc[i]['id']
        attrition = result.iloc[i]['attrition']
        rec = result.iloc[i]['recommendations']
        try:
            q = prediction_table.update().where(prediction_table.c.id == indx).values({
            'attrition' : bool(attrition),
            'recommendations': rec,
            'is_sent_to_ml': True,
            'updated_by': 'machine learning',
            'updated_time' : pd.to_datetime(datetime.now())
            })
            conn.execute(q)
        except:
            pass

    end_time = time.time()
    db_update_time = end_time - start_time
    message = 'SUCCESS'
    return prediction_time, recommend_time, db_update_time, message

def recommend(ml_core, train_data, predict_data):
    start_time = time.time()
    need_recommendation = predict_data[predict_data.attrition==1]
    train_data.set_index('id', inplace = True)
    train_data = train_data[train_features]
    train_data_led = ml_core.label_encoder.transform(train_data)
    recommender = Recommend(train_data_led, 'attrition')
    recommends_list = []
    for person_indx in range(need_recommendation.shape[0]):
        person = ml_core.label_encoder.transform(need_recommendation.iloc[person_indx:person_indx+1])
        recoms = recommender.recommend(person)
        analyzer = Analyze(need_recommendation.iloc[person_indx:person_indx+1], train_data[train_data.index.isin(recoms.index)])
        an = analyzer.analyze()
        recommends_list.append(an)

    predict_data['recommendations'] = 'NaN'
    predict_data.loc[predict_data.attrition==1, 'recommendations'] = recommends_list
    end_time = time.time()
    recommend_time = end_time - start_time
    return recommend_time, predict_data

@app.route('/')
def home():
    return jsonify(message = 'ml_ai')


@app.route('/train_data', methods=['GET', 'POST', 'PUT'])
def train_data_requst():
    if request.method == 'GET':
        data = pd.read_sql(appconfig.train_table_name, conn)
        data = data.to_json()
        return data
        # return jsonify(message = 'send trainging data via POST method to add them in ML DB or use PUT to deleted records')
    elif request.method == 'POST':
        data = pd.DataFrame(request.json)
        data.to_sql(appconfig.train_table_name, conn, if_exists='append')
        return jsonify(message = 'added data to training table')
    elif request.method == 'PUT':
        data = request.json
        if type(data['data']) == list:
            q = training_table.delete().where(training_table.c.EmployeeNumber.in_(data['data']))
            try:
                conn.execute(q)
                return jsonify(message = "removed data from training table")
            except:
                return jsonify(message = 'couldnt delete train data')
        else:
            return jsonify(message = 'unsupported data type')
    else:
        return jsonify(message = 'unsupported method'), 400


@app.route('/predict_data', methods=['GET', 'POST', 'PUT'])
def predict_data_requst():
    if request.method == 'GET':
        data = pd.read_sql(appconfig.predict_table_name, conn)
        data = data.to_json()
        return data
        # return jsonify(message = 'send trainging data via POST method to add them in ML DB or use PUT to deleted records')
    elif request.method == 'POST':
        data = pd.DataFrame(request.json)
        data.to_sql(appconfig.predict_table_name, conn, if_exists='append')
        return jsonify(message = 'added data to prediction table')
    elif request.method == 'PUT':
        data = request.json
        if type(data['data']) == list:
            q = prediction_table.delete().where(prediction_table.c.EmployeeNumber.in_(data['data']))
            try:
                conn.execute(q)
                return jsonify(message = "removed data from prediction table")
            except:
                return jsonify(message = 'couldnt delete prediction data'), 500
        else:
            return jsonify(message = 'unsupported data type'), 400
    else:
        return jsonify(message = 'unsupported method'), 400


@app.route('/predict')
def predict_reques():
    # predict_data = pd.read_sql(f"select * from {predict_table_name} where is_sent_to_ml='FALSE'", db)
    predict_data = pd.read_sql(f"select * from {predict_table_name} where is_delete=0", conn)
    # train_data = pd.read_sql(f"select * from {train_table_name} where is_sent_to_ml='FALSE'", db)
    train_data = pd.read_sql(f"select * from {train_table_name} where is_delete=0", conn)
    if predict_data.shape[0]>0:
        prediction_time, recommend_time, db_update_time, message = predict(predict_data,train_data )
        prediction_time += recommend_time + db_update_time
        status = 200
        if message == 'FAILURE':
            status = 400
        return jsonify(prediction_time = '%.2f' % prediction_time, message = message), status
    else:
        return jsonify(message = 'nothing to predict'), 400

# if __name__ == '__main__':
#     app.run(debug=True)