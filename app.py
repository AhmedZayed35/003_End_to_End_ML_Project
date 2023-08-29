import numpy as np
import pandas as pd

from flask import Flask, render_template, request

from src.logger import logging
from src.pipelines.predict_pipeline import PredictPipeline, CustomData

# Create flask app
app = Flask(__name__)

# home page
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    '''Predicts the target variable for a given data point then returns the prediction to the user in the home page'''
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender= request.form.get('gender'),
            race_ethnicity= request.form.get('race_ethnicity'),
            parental_level_of_education= request.form.get('parental_level_of_education'),
            lunch= request.form.get('lunch'),
            test_preparation_course= request.form.get('test_preparation_course'),
            reading_score= request.form.get('reading_score'),
            writing_score= request.form.get('writing_score')
        )
        
        data_df = data.get_data_as_dataframe()
        logging.info(f'Predicting for data point: {data_df.to_dict()}')
        
        prediction_pipeline = PredictPipeline()
        prediction = prediction_pipeline.predict(data_df)
        
        return render_template('home.html', results=prediction)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
        