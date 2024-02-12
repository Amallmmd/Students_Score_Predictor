import os
import sys
import numpy as np
import pandas as pd
import dill
import pickle
from pathlib import Path
from src.exception import CustomException
from src.logger import logging

def save_obj(file_path,obj):
    try:
        dir_path = Path(os.path.dirname(file_path))
        dir_path.mkdir(parents=True,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:

        raise CustomException(e,sys)