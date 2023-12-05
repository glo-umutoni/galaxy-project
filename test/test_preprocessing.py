import os
import pandas as pd
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

test_dataset_path = "test/data/test_dataset.csv"

class TestNormalize:
    ''' Test the normalize function from the Preprocessing module'''

    def test_normalize_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            # try to input object that is not the Data class object
            Preprocessing.normalize(pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}))

    def test_normalize_return_value(self):
        '''Check that None is returned'''
        data = Data()
        data.extract_from_file(test_dataset_path)
        assert Preprocessing.normalize(data) is None

    def test_normalize_correct_values(self):
        '''Compare module normalization and sklearn normalization values'''
        # normalize data with our module
        data = Data()
        data.extract_from_file(test_dataset_path)
        Preprocessing.normalize(data)
        # normalize data with sklearn
        df = pd.read_csv(test_dataset_path)
        sk_preprocess_data = preprocessing.normalize(df, norm='l2')
        # check np docs for asserting equal 
        assert sk_preprocess_data.equals(data.data)


class TestOutlierRemoval:
    ''' Test the outlier removal function from the Preprocessing module '''

    def test_remove_outliers_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            # try to input object that is not the Data class object
            Preprocessing.remove_outliers(pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}))

    def test_remove_outliers_return_value(self):
        '''Check that None is returned'''
        data = Data()
        data.extract_from_file(test_dataset_path)
        assert Preprocessing.remove_outliers(data) is None

    def test_remove_outliers_correct_values(self):
        '''Check that correct values are returned'''
        # remove outliers with our module
        data = Data()
        data.extract_from_file(test_dataset_path)
        Preprocessing.remove_outliers(data)
        # remove outliers locally
        df = pd.read_csv(test_dataset_path)
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

    def test_interpolate_return_value(self):
        '''Check that None is returned'''
        x = np.linspace(0,5,50)
        y = 3*(x**3)
        assert Preprocessing.interpolate(x, y,(2,3),20) is None

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
            Preprocessing.correct_redshift(pd.DataFrame({'c1': [1, 2], 'c2': [3, 4]}))

    def test_correct_redshift_no_column(self):
        '''Check that exception is raised when "loglam" column DNE'''
        data = Data()
        data.extract_from_file(test_dataset_path)
        data = data.drop(columns = 'loglam')
        with pytest.raises(AttributeError):
            Preprocessing.correct_redshift(data)

    def test_correct_redshift_return_value(self):
        '''Check that None is returned'''
        data = Data()
        data.extract_from_file(test_dataset_path)
        assert Preprocessing.correct_redshift(data) is None

    def test_correct_redshift_correct_values(self):
        '''Check that the redshift is calculated correctly'''
        # correct redshift according to our module
        data = Data()
        data.extract_from_file(test_dataset_path)
        Preprocessing.correct_redshift(data)
        # correct redshift locally
        df = pd.read_csv(test_dataset_path)
        redshift_cor_data = df["loglam"] - np.log(1 + df["redshift"])
        assert data["corrected_loglam"] == redshift_cor_data