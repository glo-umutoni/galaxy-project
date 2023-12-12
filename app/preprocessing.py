'''Module used to preprocess astronomical data from Sloan Digital Sky Survey.'''
import pandas as pd
from typing import Tuple
from data_extraction import Data
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from scipy import stats

class Preprocessing:
    '''Preprocesses SDSS data.'''

    @staticmethod
    def normalize(data:pd.DataFrame): 
        ''' Normalize the values in data.
        Returns the normalized values as a pandas DataFrame.

        Parameters
        --------
        data: pd.DataFrame
            DataFrame containing spectrum data from SDSS

        Returns
        -------
        a DataFrame with normalized spectrum data

        Raises
        -------
        ValueError
            Raised if data is not a pandas DataFrame
            Raised if data is empty
        '''
        if not isinstance(data, pd.DataFrame): 
            raise ValueError("data is not a pd.DataFrame")
        if data.empty: 
            raise ValueError("data is empty, make sure you extract data first.")
        scaler = StandardScaler()
        data = pd.DataFrame(scaler.fit_transform(data), columns = data.columns)
        return data

    @staticmethod
    def remove_outliers(data:pd.DataFrame):
        ''' Removes the outliers in data.
        Returns a new DataFrame with the outliers removed.

        Parameters
        --------
        data: pd.DataFrame
            DataFrame containing spectrum data from SDSS

        Returns
        -------
        a DataFrame with the outliers removed

        Raises
        -------
        ValueError
            Raised if data is empty
        '''
        if data.empty: 
            raise ValueError("data.data is empty, make sure you extract data first.")
        return data[(np.abs(stats.zscore(data)) < 2).all(axis=1)]

    @staticmethod
    def interpolate(x:list, y:list, x_lim:Tuple[float, float], bins:int):
        ''' Interpolates wavelengths for a specified object id.

        Parameters
        --------
        x: list
            A list of log wavelengths
        y: list
            A list of flux values
        x_lim: Tuple
            Contains [x_min, x_max] to interpolate between
        bins: 
            Number of points to interpolate

        Returns
        -------
        x: list
            linearly spaced log wavelengths
        y: list
            interpolated flux based on the user selected range and number of points

        Raises
        -------
        ValueError
            Raised if the bin value is not an integer
            Raised if min or max values are empty
            Raised if the shape of x and y is not (1,1)
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
        ''' Corrects the redshift in the SDSS spectral data.
        Returns a new DataFrame with a new column "corrected_loglam" \
            containing the redshift corrected values.

        Parameters
        --------
        redshift: float
            The redshift value of the object
        data: pd.DataFrame
            DataFrame containing spectrum data from SDSS. 
            Needs to contain "loglam" column.

        Returns
        -------
        a DataFrame with a new column "corrected_loglam" \
            containing the redshift corrected values

        Raises
        -------
        ValueError
            Raised if data is empty
            Raised if redshift is None
        AttributeError
            Raised if "loglam" column is not in data
        '''

        if data.empty: raise ValueError("data.data is empty, make sure you extract data first.")
        if "loglam" not in data.columns:
            raise AttributeError("loglam column doesn't exist, please double check your query.")
        if redshift is None: 
            raise ValueError("redshift value is None, please check your input values")
        
        wavelengths = data["loglam"]
        data["corrected_loglam"] = wavelengths - np.log(1 + redshift)
        return data
