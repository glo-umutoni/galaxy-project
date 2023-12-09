import numpy as np
import sys
sys.path.append("app/")
from data_extraction import Data
from preprocessing import Preprocessing
query="SELECT TOP 10 * FROM SpecObj"

class TestIntegrationNormalize:

    def test_normalize_data(self):
        data = Data()
        data.extract_from_query(query)
        data.get_spectra()
        std_data = Preprocessing.normalize(data.spectrum[0][1].data)
        assert np.testing.assert_array_almost_equal(np.mean(std_data, axis=1), np.zeros(np.mean(std_data, axis=1).shape), decimal=6)
        assert np.testing.assert_array_almost_equal(np.std(std_data, axis=1), np.ones(np.mean(std_data, axis=1).shape), decimal=6)


class TestIntegrationRemoveOutliers:

    def test_remove_outliers_data(self):
        data = Data()
        data.extract_from_query(query)
        data.get_spectra()
        clean_data = Preprocessing.remove_outliers(data.spectrum[0][1].data)
        assert clean_data.shape[1]==data.spectrum[0][1].data.shape[1]
        assert clean_data.shape[0]<=data.spectrum[0][1].data.shape[0]

class TestIntegrationCorrectRedshift:

    def test_correct_redshift_data(self):
        data = Data()
        data.extract_from_query(query)
        data.get_spectra()
        clean_data = Preprocessing.correct_redshift(data.spectrum[0][1].data, redshift=10)
        assert "corrected_loglam" in clean_data