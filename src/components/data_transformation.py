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
from src.logger import logging

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
        
    def create_transformer_obj(self):
        try:
            numerical_features = ['reading score', 'writing score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]
            
            
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)
            