# Basic Import

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
# from catboost import CatBoostRegressor
from xgboost import XGBRegressor
import warnings
import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_obj,evaluate


@dataclass
class ModelTrainerConfig:
    model_train_path = os.path.join('artifacts','model.pkl')
class ModelTrainer:
    def __init__(self):
        self.model_train_config = ModelTrainerConfig()
    def initiate_model_train(self,train_arr,test_arr):
        try:
            logging.info('Starting train test splitting')
            xtrain, ytrain, xtest, ytest = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            models = {
                'Random forest':RandomForestRegressor(),
                'Ada boost':AdaBoostRegressor(),
                'Gradient Boost':GradientBoostingRegressor(),
                'Linear Regression':LinearRegression(),
                'KNN':KNeighborsRegressor(),
                'Decision Tree':DecisionTreeRegressor(),
                'XGboost':XGBRegressor(),


            }
            param_grid = {
            'Random forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Ada boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0]
            },
            'Gradient Boost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'max_depth': [3, 5, 10]
            },
            'Linear Regression': {
                'fit_intercept': [True, False]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance']
            },
            'Decision Tree': {
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'XGboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'max_depth': [3, 5, 10]
            }
        }

            model_result:dict = evaluate(xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest,models=models,param_grid=param_grid)
            best_model_score = max(sorted(model_result.values()))
            best_model_name = list(model_result.keys())[
                list(model_result.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score < 0.6:
                raise CustomException('No good models are available')
            logging.info('Best model on Training and Testing data')
            save_obj(
                file_path = self.model_train_config.model_train_path,
                obj = best_model
            )
            predicted = best_model.predict(xtest)
            r2_square = r2_score(ytest,predicted)

            return r2_square
        except Exception as e:
            raise CustomException(e,sys)
            
        