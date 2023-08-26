import os
import sys
import dill 
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

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
    

def evaluate_models(X_train, X_test, y_train, y_test, models):
    try:
        train_report = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])
        test_report = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])
        
        for i in models.keys():
            model = models[i]
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mse = mean_absolute_error(y_train, y_train_pred)
            train_r2 = r2_score(y_train, y_train_pred)
            
            y_test_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
            test_mae = mean_absolute_error(y_test, y_test_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            
            train_report.loc[train_report.shape[0]] = {'Model': i, 'RMSE': train_rmse, 'MAE': train_mse, 'R2': train_r2}
            test_report.loc[test_report.shape[0]] = {'Model': i, 'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2} 
            # test_report = test_report.append({'Model': i, 'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2}, ignore_index=True)
            # train_report = train_report.append({'Model': i, 'RMSE': train_rmse, 'MAE': train_mse, 'R2': train_r2}, ignore_index=True)
            
        return train_report, test_report
            
    except Exception as e:
        logging.error(e)
        raise CustomException(e, sys)