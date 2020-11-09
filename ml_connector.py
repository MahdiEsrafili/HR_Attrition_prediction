from ml_core import ML_core
from Recommendation import Recommend
from recommend_analyzer import Analyze
import pandas as pd
from kafka import KafkaProducer
from kafka import KafkaConsumer
from io import StringIO
import time
import logging

logging.basicConfig(filename='ml_connector.log', filemode='a', format='%(name)s - %(levelname)s - %(message)s', level = logging.INFO)

consumer = KafkaConsumer(
    'ml_*',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group')

for message in consumer:
    topic = message.topicrr
    _, company, field = topic.split('_')
    if field=='trainData':
        # create new ml_core class and train it
        pass
    elif field=='predictData':
        # load model and predict
        pass
    else:
        pass 

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
    logging.info(f'recommend at {end_time - start_time:.3f}')
    return result
    

