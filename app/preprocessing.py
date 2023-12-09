import pandas as pd
from typing import Tuple
from data_extraction import Data
from sklearn import preprocessing
import pandas as pd
import numpy as np
from scipy import stats

class Preprocessing:

    @staticmethod
    def normalize(data:pd.DataFrame): 
        if not isinstance(data, pd.DataFrame): raise ValueError("data is not a pd.DataFrame")
        if data.empty: raise ValueError("data is empty, make sure you extract data first.")
        data = pd.DataFrame(preprocessing.normalize(data, norm='l2'))
        return data

    @staticmethod
    def remove_outliers(data:pd.DataFrame):
        if data.empty: raise ValueError("data.data is empty, make sure you extract data first.")
        return data[(np.abs(stats.zscore(data)) < 2).all(axis=1)]

    @staticmethod
    def interpolate(x:list, y:list, Range:Tuple[float, float], bin:int):
        if len(x) == 0: raise ValueError("x is empty, please pass in a non empty list.")
        if len(y) == 0: raise ValueError("y is empty, please pass in a non empty list.")
        if not isinstance(Range, Tuple): raise ValueError("Range needs to be a Tuple")
        if len(Range) != 2: raise ValueError("Range needs to have 2 values (min, max).")
        if bin <= 0: raise ValueError("bin needs to be a value greater than 0.")
        pass 


    @staticmethod
    def correct_redshift(redshift:float, data:pd.DataFrame):
        if data.empty: raise ValueError("data.data is empty, make sure you extract data first.")
        if "loglam" not in data.columns:
            raise AttributeError("loglam column doesn't exist, please double check your query.")
        if "reshift" is None: 
            raise ValueError("redshift value is None, please check your input values")
        
        wavelengths = data["loglam"]
        data["corrected_loglam"] = wavelengths - np.log(1 + redshift)
        return data

