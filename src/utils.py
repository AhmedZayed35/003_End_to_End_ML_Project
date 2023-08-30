import os
import sys
import dill 
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV

# from src.logger import logging
from src.exception import CustomException


def save_object(obj, path):
    '''Saves the object as a pickle file'''
    
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            dill.dump(obj, f)
            
        # logging.info(f'Object saved at {path}')
        
    except Exception as e:
        # logging.error(e)
        raise CustomException(e, sys)
    
    
def load_object(path):
    '''Loads the object from the pickle file'''
    try:
        with open(path, 'rb') as f:
            obj = dill.load(f)

        return obj
    
    except Exception as e:
        # logging.error(e)
        raise CustomException(e, sys)
    

def evaluate_models(X_train, X_test, y_train, y_test, models):
    '''Evaluates the models and returns the train and test reports'''
    try:
        train_report = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])
        test_report = pd.DataFrame(columns=['Model', 'RMSE', 'MAE', 'R2'])
        models_list = []
        
        for model_name, model, params in models:
            
            grid_search = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, n_jobs=-1, verbose=1)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            train_predictions = best_model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
            train_mse = mean_absolute_error(y_train, train_predictions)
            train_r2 = r2_score(y_train, train_predictions)
            
            test_predictions = best_model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_mae = mean_absolute_error(y_test, test_predictions)
            test_r2 = r2_score(y_test, test_predictions)
            
            train_report.loc[train_report.shape[0]] = {'Model': model_name, 'RMSE': train_rmse, 'MAE': train_mse, 'R2': train_r2}
            test_report.loc[test_report.shape[0]] = {'Model': model_name, 'RMSE': test_rmse, 'MAE': test_mae, 'R2': test_r2}
            models_list.append((model_name, best_model, grid_search.best_params_))
            
            # logging.info(f'Evaluation completed for {model_name}')
           
        # logging.info('Evaluation completed for all models')
        return train_report, test_report, models_list
            
    except Exception as e:
        # logging.error(e)
        raise CustomException(e, sys)
    
def get_best_model_obj(models_list, target_model_name):
    '''Returns the best model object from the list of models'''
    for model_name, model_obj, _ in models_list:
        if model_name == target_model_name:
            return model_obj
        
