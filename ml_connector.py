from ml_core import ML_core
import pandas as pd
from kafka import KafkaProducer
from kafka import KafkaConsumer
from io import StringIO


consumer = KafkaConsumer(
    'ml_*',
     bootstrap_servers=['localhost:9092'],
     auto_offset_reset='earliest',
     enable_auto_commit=True,
     group_id='my-group')

for message in consumer:
    topic = message.topic
    _, company, field = topic.split('_')
    if field=='trainData':
        # create new ml_core class and train it
        pass
    elif field=='predictData':
        # load model and predict
        pass
    else:
        pass 

# train_data = pd.read_csv('data/name_attached/train_data_incl_names.csv')
# predict_data = pd.read_csv('data/name_attached/prediction_data_incl_names.csv') 
# ml_core = ML_core(train_data, predict_data)
# ml_core.train()
# result = ml_core.predict()
# result.to_csv('runs/prediction_result.csv')