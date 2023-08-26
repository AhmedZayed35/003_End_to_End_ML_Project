import sys
import os
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# create data ingestion class
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()
        
    def initialize_data_ingestion(self):
        logging.info("Initializing data ingestion")
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            os.makedirs('artifacts', exist_ok=True)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            df.to_csv(self.config.raw_data_path, index=False, header=True)
            train_set.to_csv(self.config.train_data_path, index=False, header=True)
            test_set.to_csv(self.config.test_data_path, index=False, header=True)
            
            logging.info("Data ingestion completed")
            
            return(
                self.config.train_data_path, 
                self.config.test_data_path, 
                )
            
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initialize_data_ingestion()
    
    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(X_train, X_test, y_train, y_test))
    