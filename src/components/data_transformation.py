import sys
import os
import numpy as np
import pandas as pd

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
# from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        
    def create_transformer_obj(self):
        '''Creates a transformer object for the data transformation pipeline'''
        
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            # logging.info('Created Pipeline object for categirical features')
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('onehot', OneHotEncoder(handle_unknown='ignore'))
                ]
            )
            # logging.info('Created Pipeline object for categirical features')
            
            preprocess_pipeline = ColumnTransformer(
                [
                    ('num', num_pipeline, numerical_features),
                    ('cat', cat_pipeline, categorical_features)
                ]
            )
            # logging.info('Created ColumnTransformer object')
            
            return preprocess_pipeline
            
        except Exception as e:
            # logging.error(e)
            raise CustomException(e, sys)
        
        
    def initiate_data_transformation(self, train_path, test_path):
        '''Initiates the data transformation process'''
        
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            # logging.info('Loaded train and test data')
            
            preprocess_pipeline = self.create_transformer_obj()
            # logging.info('Created transformer object')
            
            target = 'math_score'
            X_train = preprocess_pipeline.fit_transform(train_data.drop(target, axis=1))
            X_test = preprocess_pipeline.transform(test_data.drop(target, axis=1))
            y_train = train_data[target]
            y_test = test_data[target]
            # logging.info('Transformed train and test data')            
            
            save_object(preprocess_pipeline, self.config.preprocessor_obj_path)
            # logging.info('Saved transformer object')
            
            
            return X_train, X_test, y_train, y_test, self.config.preprocessor_obj_path
        except Exception as e:
            # logging.error(e)
            raise CustomException(e, sys)