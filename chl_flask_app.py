from flask import Flask, request, jsonify, redirect, url_for, send_from_directory
import time
from werkzeug.utils import secure_filename
import os
upload_dir = 'upload_dir'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_dir

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
        filename = secure_filename(predict_file.filename)
        predict_file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # return redirect(url_for('uploaded_file', filename=filename))
        return jsonify(message = ' trained')


if __name__ == '__main__':
    app.run(debug=True)