import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation, DataTransformationConfig

@dataclass
class DataInjestionConfig:
    train_path: str = os.path.join('artifacts','train.csv')
    test_path: str = os.path.join('artifacts', 'test.csv')
    raw_path: str = os.path.join('artifacts','raw.csv')
class DataInjestion:
    def __init__(self):
        self.injestion_config = DataInjestionConfig()
        '''
        here we create an object variable to store all the path variables used in the
        class -> DataInjestionConfig

        '''
    def inilializing_ingestion(self):
        '''
        Here we initiate the ingestion part and also make a new directory
        '''
        logging.info('Build data ingestion method')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('Succesfully read the dataset')
            directory = Path(os.path.dirname(self.injestion_config.train_path))
            directory.mkdir(parents=True, exist_ok=True)
            # make a folder and extracts the directory path from the full path stored in self.injestion_config.train_path
            # exist_ok ->if the directory already exists, no exception is raised 
            df.to_csv(self.injestion_config.raw_path,index=False,header=True)
            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(df, test_size=0.2,random_state=2)

            train_set.to_csv(self.injestion_config.train_path,index = False, header = True)
            test_set.to_csv(self.injestion_config.test_path, index = False, header = True)
            logging.info('Data ingestion has completed')

            return(
                self.injestion_config.train_path,
                self.injestion_config.test_path
            )
        except Exception as e:
            raise CustomException(e,sys)
if __name__ == '__main__':
    obj = DataInjestion()
    train_data,test_data =  obj.inilializing_ingestion() # when initiating this method using obj it will return train data and test data
    data_transformations = DataTransformation()
    data_transformations.initiate_data_transformation(train_data,test_data)