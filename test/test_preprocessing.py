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

class TestPreprocessing:
    ''' Test the Preprocessing class from the preprocessing module '''

    def test_normalization(self):
        preprocessing = Preprocessing()
        data = Data()
        sk_preprocess_data = preprocessing.normalize(data.data, norm='l2')
        assert preprocessing.normalize(data) == sk_preprocess_data

    def test_outlier_removal(self):
        preprocessing = Preprocessing()
        data = Data()
        data_no_outliers = data.data[(np.abs(stats.zscore(data.data)) < 2).all(axis=1)]
        assert preprocessing.normalize(data) == data_no_outliers

    def test_interpolate(self):
        # write test
        pass

    def test_redshift_correction(self):
        data = Data()
        redshift_cor_data = data.data["loglam"] - np.log(1 + data.data["redshift"])
        assert preprocessing.redshift_correction(data) == redshift_cor_data