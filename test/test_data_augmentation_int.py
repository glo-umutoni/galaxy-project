'''Test integration of data_augmentation module'''

import numpy as np
from astroquery.sdss import SDSS
import sys
import pandas as pd
sys.path.append("app/patapoufsinthestars")
from data_extraction import Data
from preprocessing import Preprocessing
from data_augmentation import DataAugmentor

query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
new_dtype = np.dtype([('flux', '<f4'), ('loglam', '<f4'), ('ivar', '<f4'),
                            ('and_mask', '<i4'), ('or_mask', '<i4'),
                            ('wdisp', '<f4'), ('sky', '<f4'), ('model', '<f4')])

class TestIntegrationDataAugmentation:


    def test_data_augmentation_derivative_for_one_unaligned_obj_with_preprocessing(self):
        '''Compute derivative of unaligned spectra for one star'''   
        # extract data from data_extraction
        data = Data()
        data.extract_from_query(query)
        spectra = data.get_spectra_from_data()
        unaligned_spectra = pd.DataFrame(spectra[0][1].data)

        # preprocess unaligned spectra
        unaligned_spectra = Preprocessing.correct_redshift(redshift = 10, data = unaligned_spectra)
        unaligned_spectra_std = Preprocessing.normalize(unaligned_spectra)
       
        unaligned_deriv = DataAugmentor.compute_derivative(unaligned_spectra_std, derivative_order=[1])

        assert unaligned_deriv.shape == (unaligned_spectra_std.shape[0], 1+unaligned_spectra_std.shape[1])



    def test_data_augmentation_frac_derivative_for_single_unaligned_obj_with_preprocessing(self):
        '''Compute fractional derivative of unaligned spectra for one star'''

        # extract data from data_extraction
        data = Data()
        data.extract_from_query(query)
        spectra = data.get_spectra_from_data()
        unaligned_spectra = pd.DataFrame(spectra[0][1].data)

        # preprocess unaligned spectra
        unaligned_spectra = Preprocessing.correct_redshift(redshift = 10, data = unaligned_spectra)
        unaligned_spectra_std = Preprocessing.normalize(unaligned_spectra)
       
        unaligned_deriv = DataAugmentor.compute_derivative(unaligned_spectra_std, derivative_order=[0.7])

        assert unaligned_deriv.shape == (unaligned_spectra_std.shape[0], 1+unaligned_spectra_std.shape[1])
