import os
import sys
import dill 
import pandas as pd
import numpy as np

from src.logger import logging
from src.exception import CustomException

def save_object(obj, path):
    '''Saves the object as a pickle file'''
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            dill.dump(obj, f)
            
        logging.info(f'Object saved at {path}')
        
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)