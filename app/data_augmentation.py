'''Module used to compute derivatives as well as fractional derivatives'''
import sys
sys.path.append("app/")
from differint.differint import GL
import numpy as np

class DataAugmentor:
    @staticmethod
    def compute_derivative(data:np.ndarray, derivative_order=[0.5,1]):
        ''' Computes derivatives as well as fractional derivatives

         Parameters
         ----------
         data: list, or 1d-array of  function values
         derivative_order: list The order of the differintegral to be computed. Default values is [0.5, 1]

         Returns
         --------
         augment_data: np.array of augmented data of shape (o, s, f) where:
             o: number of derivatives
             s: number of unique star object ids
             f: number of flux points entered

         Raises
         --------
         ValueError
             Raised if data type is not list or ndarray
             Raised if the derivative order is not a list

         '''
        if data is None:
            raise ValueError("Data to derive cannot be None")
        if not isinstance(data, np.ndarray):
            raise ValueError("Data to derive must be a np.ndarray")
        if not len(data.shape) != 2 :
            raise ValueError("Data to derive must be a 2D array")
        if None in derivative_order:
            raise ValueError("Derivative order cannot be None")
        if type(derivative_order) != list:
            raise ValueError("Derivative order must be a list")
        # store all the derived data, one list per order of derivaive
        augmented_data= []
        for alpha in derivative_order:
            derived_data= np.apply_along_axis(lambda row: GL(alpha=alpha, f_name=row, num_points=len(row)), axis=1, arr=data)
            augmented_data.append(derived_data)
        return np.array(augmented_data)