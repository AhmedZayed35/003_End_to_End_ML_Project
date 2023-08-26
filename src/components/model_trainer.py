import os
import sys

from dataclasses import dataclass

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def initiate_model_training(self, X_train, X_test, y_train, y_test):
        try:
            logging.info('Initializing model training')
            models = {
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'AdaBoostRegressor': AdaBoostRegressor(),
                'LinearRegression': LinearRegression(),
                'DecisionTreeRegressor': DecisionTreeRegressor(),
                'KNeighborsRegressor': KNeighborsRegressor(),
                'XGBRegressor': XGBRegressor(),
                'CatBoostRegressor': CatBoostRegressor(verbose=False)
            }
            
            _, test_report = evaluate_models(X_train= X_train, X_test= X_test, y_train= y_train, y_test= y_test, models= models)
            logging.info(f"models test report: {test_report.to_dict()}")
            
            best_model = test_report.sort_values(by='R2', ascending=False).iloc[0]
            
            if best_model['R2'] < 0.6:
                raise CustomException('No model has R2 score greater than 0.6', sys)
            
            logging.info(f'Best model is {best_model["Model"]} with R2 score of {best_model["R2"]}')
            
            model = models[best_model['Model']]
            
            save_object(model, self.config.model_path)
            logging.info(f'Model saved at {self.config.model_path}')
            
            return r2_score(y_test, model.predict(X_test))
            
            
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

