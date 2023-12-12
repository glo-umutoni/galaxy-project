import os
import pandas as pd
from astroquery.sdss import SDSS
import shutil
import pytest
import sys
import numpy as np
from scipy import stats
from sklearn import preprocessing
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
sys.path.append("app/")
from data_extraction import Data
from preprocessing import Preprocessing


query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
new_dtype = np.dtype([('flux', '<f4'), ('loglam', '<f4'), ('ivar', '<f4'),
                            ('and_mask', '<i4'), ('or_mask', '<i4'),
                            ('wdisp', '<f4'), ('sky', '<f4'), ('model', '<f4')])

class TestNormalize:
    ''' Test the normalize function from the Preprocessing module'''

    def test_normalize_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            Preprocessing.normalize(pd.DataFrame({}))

    def test_normalize_return_value(self):
        '''Check that a pd.DataFrame is returned'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        data = pd.DataFrame(data.values, columns=data.columns, index=data.index)
        assert isinstance(Preprocessing.normalize(data), pd.DataFrame)

    def test_normalize_correct_values(self):
        ''' Check that normalize function actually returns a normalized array'''
        # normalize data with our module
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        new_data = Preprocessing.normalize(data)

        assert np.sum(np.mean(new_data, axis=0))<10**(-6)
        assert np.sum(np.std(new_data, axis=0))-len(np.mean(new_data, axis=0))<10**(-6)
 

class TestOutlierRemoval:
    ''' Test the outlier removal function from the Preprocessing module '''

    def test_remove_outliers_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            Preprocessing.remove_outliers(pd.DataFrame({}))

    def test_remove_outliers_return_value(self):
        '''Check that a pd.DataFrame is returned'''
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)

        # Copy values from the original array to the new array
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]

        data = pd.DataFrame(new_array)
        assert isinstance(Preprocessing.remove_outliers(data), pd.DataFrame)

    def test_remove_outliers_correct_values(self):
        '''Check that correct values are returned'''
        # remove outliers with our module
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)

        # Copy values from the original array to the new array
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]

        df = pd.DataFrame(new_array)
        new_df = Preprocessing.remove_outliers(df)
        # remove outliers locally
        data_no_outliers = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
        assert data_no_outliers.equals(new_df)


class TestInterpolate:
    ''' Test the interpolation function from the Preprocessing module '''

    def test_interpolate_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        two_dim_arr = np.array([[2,3],[3,4]])
        one_dim_arr = np.linspace(0,1,6)

        with pytest.raises(ValueError):
            # try to input a 2d array as x
            Preprocessing.interpolate(two_dim_arr, one_dim_arr, (0,1), 10)
            # try to input a 2d array as y
            Preprocessing.interpolate(one_dim_arr, two_dim_arr, (0,1), 10)
            # wrong range input
            Preprocessing.interpolate(one_dim_arr, one_dim_arr, (1,0), 10)
            Preprocessing.interpolate(one_dim_arr, one_dim_arr, 1, 10)
            # wrong num_datapoints
            Preprocessing.interpolate(one_dim_arr, one_dim_arr, (0,1), 5.5)

    def test_interpolate_correct_values(self):
        '''Test interpolation function returns correct values'''
        x = np.linspace(0,5,50)
        x_new = np.linspace(2,3,20)
        y = 3*(x**3)
        interp_fn = interp1d(x,y,kind='linear', fill_value='extrapolate')
        assert np.array_equal(interp_fn(x_new),Preprocessing.interpolate(x,y,(2,3),20)[1])
        assert np.array_equal(x_new,Preprocessing.interpolate(x,y,(2,3),20)[0])


class TestCorrectRedshift:
    ''' Test the redshift correct function from the Preprocessing module '''

    def test_correct_redshift_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            Preprocessing.correct_redshift(redshift = 10, data=pd.DataFrame({}))

    def test_correct_redshift_no_column(self):
        '''Check that exception is raised when "loglam" column DNE'''
        # Assuming SDSS.get_spectra(matches=query_result)[0][1].data is your DataFrame
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)

        # Copy values from the original array to the new array
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]

        df= pd.DataFrame(new_array)
        df = df.drop(columns='loglam')
        with pytest.raises(AttributeError):
            Preprocessing.correct_redshift(redshift = 10, data=df)

    def test_correct_redshift_return_value(self):
        '''Check that a pd.DataFrame is returned'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        assert isinstance(Preprocessing.correct_redshift(redshift = 10, data=data), pd.DataFrame)

    def test_correct_redshift_correct_values(self):
        '''Check that the redshift is calculated correctly'''
        # correct redshift according to our module
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        new_data = Preprocessing.correct_redshift(redshift = 10, data=data)
        # correct redshift locally
        df = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        redshift_cor_data = df["loglam"] - np.log(1 + 10)
        assert np.array_equal(new_data["corrected_loglam"], redshift_cor_data)