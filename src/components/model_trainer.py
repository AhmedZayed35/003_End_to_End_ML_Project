import os
import sys

from dataclasses import dataclass

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models, get_best_model_obj

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Initializing model training')
            models = [
                ('Random Forest Regressor', RandomForestRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 8, 15, 25, 30, None],
                    'max_features': ['log2', 'sqrt', None],
                    'random_state': [42],
                    'n_jobs': [-1]
                }),
                ('Gradient Boosting Regressor', GradientBoostingRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 8, 15, 25, 30, None],
                    'max_features': ['log2', 'sqrt', None],
                    'random_state': [42],
                    'learning_rate': [0.01, 0.05, 0.1]
                }),
                ('Ada Boost Regressor', AdaBoostRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'loss': ['linear', 'square', 'exponential'],
                    'random_state': [42],
                }),
                ('Linear Regression', LinearRegression(), {}),
                ('Decision Tree Regressor', DecisionTreeRegressor(), {
                    'criterion': ['mse', 'friedman_mse', 'mae'],
                    'splitter': ['best', 'random'],
                    'max_depth': [5, 8, 15, 25, 30, None],
                    'max_features': ['log2', 'sqrt', None],
                    'random_state': [42]
                }),
                ('KNeighbors Regressor', KNeighborsRegressor(), {
                    'n_neighbors': [3, 5, 10],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree'],
                }),
                ('XGBRegressor', XGBRegressor(), {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [5, 8, 15, 25, 30, None],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'booster': ['gbtree', 'gblinear', 'dart'],
                    'random_state': [42]
                }),
                ('Cat Boost Regressor', CatBoostRegressor(), {
                    'iterations': [100, 200, 300],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [3, 5, 8],
                    'loss_function': ['RMSE', 'MAE'],
                    'random_seed': [42]
                })
            ]
            
            _, test_report, trained_models_list = evaluate_models(X_train= X_train, X_test= X_test, y_train= y_train, y_test= y_test, models= models)
            logging.info(f"models test report: {test_report.to_dict()}")
            
            best_model_report = test_report.sort_values(by='R2', ascending=False).iloc[0]
            if best_model_report['R2'] < 0.6:
                raise CustomException('No model has R2 score greater than 0.6', sys)
            logging.info(f'Best model is {best_model_report["Model"]} with R2 score of {best_model_report["R2"]}')
            
            best_model = get_best_model_obj(trained_models_list, best_model_report['Model'])
            save_object(best_model, self.config.model_path)
            logging.info(f'Model saved at {self.config.model_path}')
            
            return r2_score(y_test, best_model.predict(X_test))
            
            
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

