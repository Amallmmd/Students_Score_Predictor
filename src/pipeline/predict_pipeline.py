import sys
from src.exception import CustomException
import pandas as pd
import os
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            logging.info(f'Loading model from {model_path}')
            model = load_object(filename=model_path)
            logging.info('Model loaded successfully')

            logging.info(f'Loading preprocessor from {preprocessor_path}')
            preprocessor = load_object(filename=preprocessor_path)
            logging.info('Preprocessor loaded successfully')

            logging.info(f'Feature shape: {features.shape}')
            scaled_data = preprocessor.transform(features)
            logging.info(f'Scaled data: {scaled_data}')

            preds = model.predict(scaled_data)
            logging.info(f'Predictions: {preds}')
            return preds
        except Exception as e:
            raise CustomException(e,sys)
            
    

class CustomData:  # this class is responsible for mapping all the input values given from the html to the backend
    def __init__(self,
                 gender : str,
                race_ethnicity : str,
                parental_level_of_education : str,
                lunch : str,
                test_preparation_course : str,
                writing_score : int,
                reading_score : int
                ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.writing_score = writing_score
        self.reading_score = reading_score
        # self.math_score = math_score
    def getdata_as_dataframe(self): # this method is to convert the input data into dataframe
        try:
            custom_data_input_dict = {
                "gender" : [self.gender],
                "race_ethnicity" : [self.race_ethnicity],
                "parental_level_of_education":  [self.parental_level_of_education],
                "lunch" : [self.lunch],
                "test_preparation_course" : [self.test_preparation_course],
                "writing_score":  [self.writing_score],
                "reading_score":  [self.reading_score],
                # "math_score":  [self.math_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e,sys)