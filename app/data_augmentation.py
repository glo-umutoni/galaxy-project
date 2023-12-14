'''Module used to compute derivatives as well as fractional derivatives'''
import sys
sys.path.append("app/")
import pandas as pd
from differint.differint import GL
import numpy as np

class DataAugmentor:
    '''Class that augments the data by computing both derivatives and fractional derivatives of the data.'''
    @staticmethod
    def compute_derivative(data:pd.DataFrame, column:str="flux", derivative_order=[0.5,1]):
        ''' Computes derivatives as well as fractional derivatives

         Parameters
         ----------
         data : list, or 1d-array of  function values
         derivative_order : list The order of the differintegral to be computed. Default values is [0.5, 1]

         Returns
         --------
         augment_data : np.array of augmented data of shape (o, s, f) where:
             o : number of derivatives
             s : number of unique star object ids
             f : number of flux points entered

         Raises
         --------
         ValueError
             Raised if data type is not list or ndarray
             Raised if the derivative order is not a list

         '''
        if type(data) != pd.DataFrame:
            raise ValueError("Data should be a pd.DataFrame")
        if column not in data :
            raise ValueError("column should be in data")
        if type(derivative_order) != list:
            raise ValueError("Derivative order must be a list")
        # store all the derived data, one list per order of derivaive
        df = data.copy(deep=True)
        for alpha in derivative_order:
            df[f"{column}_{alpha}"]=pd.Series(GL(alpha=alpha, f_name=df[column].array, num_points=len(df[column].array)))
        return df