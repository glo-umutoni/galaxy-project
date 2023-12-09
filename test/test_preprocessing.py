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

class TestNormalize:
    ''' Test the normalize function from the Preprocessing module'''

    def test_normalize_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            # try to input object that is not the Data class object
            Preprocessing.normalize(pd.DataFrame({}))

    def test_normalize_return_value(self):
        '''Check that None is returned'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        assert Preprocessing.normalize(data) is None

    def test_normalize_correct_values(self):
        '''Compare module normalization and sklearn normalization values'''
        # normalize data with our module
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        Preprocessing.normalize(data)
        # normalize data with sklearn
        df = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        sk_preprocess_data = preprocessing.normalize(df, norm='l2')
        # check np docs for asserting equal 
        assert np.testing.assert_array_equal(sk_preprocess_data.equals, data.data)
 

class TestOutlierRemoval:
    ''' Test the outlier removal function from the Preprocessing module '''

    def test_remove_outliers_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            # try to input object that is not the Data class object
            Preprocessing.remove_outliers(pd.DataFrame({}))

    def test_remove_outliers_return_value(self):
        '''Check that None is returned'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        assert Preprocessing.remove_outliers(data) is None

    def test_remove_outliers_correct_values(self):
        '''Check that correct values are returned'''
        # remove outliers with our module
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        Preprocessing.remove_outliers(data)
        # remove outliers locally
        df = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        data_no_outliers = df[(np.abs(stats.zscore(df)) < 2).all(axis=1)]
        assert data_no_outliers.equals(data.data)


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
        interp_fn = interp1d(x,y,kind='linear')
        assert np.array_equal(interp_fn(x_new),Preprocessing.interpolate(x,y,(2,3),20))


class TestCorrectRedshift:
    ''' Test the redshift correct function from the Preprocessing module '''

    def test_correct_redshift_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            # try to input object that is not the Data class object
            Preprocessing.correct_redshift(redshift = 10, data=pd.DataFrame({}))

    def test_correct_redshift_no_column(self):
        '''Check that exception is raised when "loglam" column DNE'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        data = data.drop(columns = 'loglam')
        with pytest.raises(AttributeError):
            Preprocessing.correct_redshift(redshift = 10, data=data)

    def test_correct_redshift_return_value(self):
        '''Check that None is returned'''
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        assert Preprocessing.correct_redshift(redshift = 10, data=data) is None

    def test_correct_redshift_correct_values(self):
        '''Check that the redshift is calculated correctly'''
        # correct redshift according to our module
        data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        Preprocessing.correct_redshift(redshift = 10, data=data)
        # correct redshift locally
        df = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)
        redshift_cor_data = df["loglam"] - np.log(1 + 10)
        assert np.array_equal(data["corrected_loglam"], redshift_cor_data)