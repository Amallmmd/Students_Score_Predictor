import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from pathlib import Path
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path = Path(os.path.dirname(file_path))
        dir_path.mkdir(parents=True,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:

        raise CustomException(e,sys)
def evaluate(xtrain,ytrain,xtest,ytest,models):
    result = {}
    for i in range(len(list(models))):
        model = list(models.values())[i]
        model.fit(xtrain,ytrain)
        ytrain_pred = model.predict(xtrain)
        ytest_pred = model.predict(xtest)
        train_model_score = r2_score(ytrain,ytrain_pred)
        test_model_score = r2_score(ytest,ytest_pred)
        result[list(models.keys())[i]] = test_model_score
    return result
