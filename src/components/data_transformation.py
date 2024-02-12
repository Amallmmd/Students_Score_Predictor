import sys
from dataclasses import dataclass
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.impute import SimpleImputer
# imputer is used to handle missing values in every features
from sklearn.pipeline import Pipeline
#-->It is particularly useful for chaining together multiple preprocessing steps followed by a model fitting step.
#--> crucial to ensure that preprocessing steps (such as scaling or imputation) are applied separately to training and test sets. 
from sklearn.compose import ColumnTransformer
# used to give different transformations to different columns bases on their type(numerical/ categorical)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
import os
import pandas as pd

@dataclass
class DataTransformationConfig:
    preprocessing_model_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        #creating method to defing trainformation 
    def data_transformation_obj(self):
        '''
        This method is responsible for data transformation on different types of features
        '''
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
            ]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')), #TO handle the missing values with median
                    ('scaler',StandardScaler(with_mean=False)), #scaling values of numerical categories
                ]# This step need to do fit_transform for training and transform for testing
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder()), #1 hot encoding all categorical fetures 
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )# steps follows same operation as before

            logging.info(f'Categorical columns {categorical_columns}')
            logging.info(f'Numerical columns {numerical_columns}')

            #create an object to combine numerical and categorical transformation
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),# (name,pipline variable, column type)
                    ('cat_pipeline',cat_pipeline,categorical_columns), # ""
                ]
            )

            logging.info('Standard scaling of numerical features done')
            logging.info('One hot encoding of categorical features done')

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    def initiate_data_transformation(self,train_path,test_path):
        '''
        This method is handling data transformation
        '''
        logging.info('Build Data Ingestion Method')

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and test path read succesfully')
            logging.info('Obtaining preprocessin object')
            preprocessin_obj = self.data_transformation_obj()
            target_column = 'math_score'
            numerical_columns = ['reading_score','writing_score']
            inputing_train_features = train_df.drop(columns=[target_column],axis=1)
            
            target_train_features = train_df[target_column]
            inputing_test_features = test_df.drop(columns=[target_column],axis=1)
            target_test_features = test_df[target_column]
            logging.info('Applying preprocessing operations on training and tesing dataframes')
            input_train_arr = preprocessin_obj.fit_transform(inputing_train_features)
            input_test_arr = preprocessin_obj.transform(inputing_test_features)

            train_arr = np.c_[
                input_train_arr,np.array(target_train_features)
            ]
            test_arr = np.c_[
                input_test_arr,np.array(target_test_features)
    
            ]
            logging.info('Saved preprocessing object successfully')
            save_obj(
                file_path = self.data_transformation_config.preprocessing_model_path,
                obj = preprocessin_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_model_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
