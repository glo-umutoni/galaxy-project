import numpy as np
import sys
import pandas as pd
sys.path.append("app/patapoufsinthestars")
from data_extraction import Data
from preprocessing import Preprocessing
query="SELECT TOP 10 * FROM SpecObj"
obj_id = "1237645879551066262"
new_dtype = np.dtype([('flux', '<f4'), ('loglam', '<f4'), ('ivar', '<f4'),
                            ('and_mask', '<i4'), ('or_mask', '<i4'),
                            ('wdisp', '<f4'), ('sky', '<f4'), ('model', '<f4')])

class TestIntegrationNormalize:

    def test_normalize_data(self):
        data = Data()
        data.extract_from_query(query)
        df = data.get_spectra_from_data()
        df = pd.DataFrame(df[0][1].data)
        std_data = Preprocessing.normalize(data=df)
        assert np.sum(np.mean(std_data, axis=0))<10**(-6)
        assert np.sum(np.std(std_data, axis=0))-len(np.mean(std_data, axis=0))<10**(-6)


class TestIntegrationRemoveOutliers:

    def test_remove_outliers_data(self):
        data = Data()
        data.extract_from_query(query)
        df = data.get_spectra_from_data()[0][1].data
        
        new_array = np.empty(df.shape, dtype=new_dtype)

        # Copy values from the original array to the new array
        for field in df.dtype.names:
            new_array[field] = df[field]

        df = pd.DataFrame(new_array)
        clean_data = Preprocessing.remove_outliers(df)
        assert clean_data.shape[1]==df.shape[1]
        assert clean_data.shape[0]<=df.shape[0]

class TestIntegrationCorrectRedshift:

    def test_correct_redshift_data(self):
        data = Data()
        data.extract_from_query(query)
        df = data.get_spectra_from_data()
        df = pd.DataFrame(df[0][1].data)
        clean_data = Preprocessing.correct_redshift(data=df, redshift=10)
        assert "corrected_loglam" in clean_data
        
