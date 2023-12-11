import pandas as pd
from typing import Tuple
from data_extraction import Data
from sklearn import preprocessing
from scipy.interpolate import interp1d
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
    def interpolate(x:list, y:list, x_lim:Tuple[float, float], bins:int):
        ''' Interpolates wavelengths for a specified object id
        Parameters
        --------
        x: a 1-d array
        y: a 1-d array
        x_lim: a tuple (x_min, x_max) to interpolate between
        bins: Number of points to interpolate

        Returns
        -------
        x: linearly spaced log wavelengths
        y: interpolated flux based on the user selected range and number of points

        Raises
        -------
        ValueError
            Raised if more than one object id is entered
            Raised if the bin value is not an integer
            Raised if min or max values are empty
        '''

        if type(bins) != int:
            raise ValueError("Number of points to interpolate needs to be an integer")
        if x_lim[0] > x_lim[1]:
            raise ValueError("x_lin not valid")
        if len(x_lim)!=2:
            raise ValueError("x_lim should be a tuple with 2 elements")
        if len(np.array(x).shape)!=1 and np.array(x).shape[1]!=1:
            raise ValueError("x should be a 1D array")
        if len(np.array(y).shape)!=1 and np.array(y).shape[1]!=1:
            raise ValueError("y should be a 1D array")
        # Convert object id into a sql query and query Specobj table for the relevant  spectra columns
        # Interpolate using SciPy's interp1d
        interp_function = interp1d(x=x, y=y, kind='linear', fill_value='extrapolate')
        # Linearly spaced wavelengths and interpolate
        x_1 = np.linspace(x_lim[0], x_lim[1], bins)
        y_1 = interp_function(x_1)
        return x_1, y_1


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

