import sys

import pandas as pd

# from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, data_point):
        '''Predicts the target variable for a given data point'''
        
        try:
            model_path = 'artifacts/model.pkl'
            preprocessing_pipeline_path = 'artifacts/preprocessor.pkl'
            model = load_object(model_path)
            preprocessor = load_object(preprocessing_pipeline_path)
            transformed_data_point = preprocessor.transform(data_point)
            preds = model.predict(transformed_data_point)
            return preds[0]
        
        except Exception as e:
            # logging.error(e)
            raise CustomException(e, sys)
    
class CustomData:
    '''Creates a custom data point for prediction'''
    def __init__(self, 
                 gender: str,
                 race_ethnicity: str,
                 parental_level_of_education: str,
                 lunch: str,
                 test_preparation_course: str,
                 reading_score: int,
                 writing_score: int,):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_dataframe(self):
        '''Returns the data as a dataframe'''
        try:
            return pd.DataFrame({ 
                                 'gender': [self.gender],
                                 'race_ethnicity': [self.race_ethnicity],
                                 'parental_level_of_education': [self.parental_level_of_education],
                                 'lunch': [self.lunch],
                                 'test_preparation_course': [self.test_preparation_course],
                                 'reading_score': [self.reading_score],
                                 'writing_score': [self.writing_score]
                                })
        
        except Exception as e:
            # logging.error(e)
            raise CustomException(e, sys)