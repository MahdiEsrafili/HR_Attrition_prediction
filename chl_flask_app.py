from flask import Flask, request, jsonify, redirect, url_for, send_from_directory
import time
from werkzeug.utils import secure_filename
import os
import pandas as pd
from ml_core import ML_core
from Recommendation import Recommend
from recommend_analyzer import Analyze
from sqlalchemy import create_engine 
fro
impo
db_string = "postgres://mluser:123456789@localhost:5432/mldb"
train_table_name = 'train2'
predict_table_name = 'prediction_data'
db = create_engine(db_string)  

upload_dir = 'upload_dir'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_dir

def train_predict_recommend(train_data, predict_data):
    start_time = time.time()
    none_informative = ['Name', 'Unnamed: 0']
    train_data = train_data.drop(none_informative, 1)
    predict_data = predict_data.drop(none_informative, 1)
    ml_core = ML_core(train_data, predict_data)
    ml_core.train()
    result = ml_core.predict()
    train_data_led = ml_core.label_encoder.transform(train_data)
    recommender = Recommend(train_data_led, 'Attrition')
    need_recommendation = result[result.Attrition==1]
    recommends_list = []
    for person_indx in range(need_recommendation.shape[0]):
        person = ml_core.label_encoder.transform(need_recommendation.iloc[person_indx:person_indx+1])
        recoms = recommender.recommend(person)
        analyzer = Analyze(need_recommendation.iloc[person_indx:person_indx+1], train_data.iloc[recoms.index])
        an = analyzer.analyze()
        recommends_list.append(an)
        
    result['recommendations'] = 'NaN'
    result.loc[result.Attrition==1, 'recommendations'] = recommends_list
    end_time = time.time()
    ml_duration = end_time - start_time
    # logging.info(f'recommend at {ml_duration:.3f}')
    return result, ml_duration

def train(train_data):
    start_time = time.time()
    none_informative = ['Name', 'index]
    train_data = train_data.drop(none_informative, 1)
    predict_data = predict_data.drop(none_informative, 1)
    ml_core = ML_core(train_data, predict_data)
    ml_core.train()


@app.route('/')
def home():
    return jsonify(message = 'ml_ai')

@app.route('/upload_train_predict', methods = ['POST'])
def upload_train_predict():
    train_file = request.files['train_file']
    predict_file = request.files['predict_file']
    if train_file and predict_file:
        filename = secure_filename(train_file.filename)
        train_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        train_data= pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        filename = secure_filename(predict_file.filename)
        predict_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        predict_data = pd.read_csv(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        result, duration = train_predict_recommend(train_data, predict_data)
        result.to_csv(os.path.join(app.config['UPLOAD_FOLDER'], 'results.csv'))
        # return redirect(url_for('uploaded_file', filename=filename))
        return jsonify(message = ' trained', duration = duration)


@app.route('/train', methods = ['GET', 'POST'])
def train_requst():
    if request.method == 'GET':
        return jsonify(message = 'give company name and excluded features')

    if 'company_name' in request.json:
        data_psql = pd.read_sql(f"select * from '{train_table_name}' where is_sent_to_ml=FALSE", db)
        return jsonify(company_name= company_name, exclude_features = exclude_features, row_count = data_psql.shape)

    else:
        return jsonify(message = 'give company name and excluded features'), 400




if __name__ == '__main__':
    app.run(debug=True)