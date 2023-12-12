'''Test integration of data_augmentation module'''

import numpy as np
from astroquery.sdss import SDSS
import sys
import pandas as pd
sys.path.append("app/")
from data_extraction import Data
from preprocessing import Preprocessing
from wavelength_alignment import WavelengthAlignment
from data_augmentation import DataAugmentor

query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
new_dtype = np.dtype([('flux', '<f4'), ('loglam', '<f4'), ('ivar', '<f4'),
                            ('and_mask', '<i4'), ('or_mask', '<i4'),
                            ('wdisp', '<f4'), ('sky', '<f4'), ('model', '<f4')])

class TestIntegrationDataAugmentation:
    def test_augment_data_derivative_aligned_spectra(self):
        '''Compute derivatives of aligned spectra for multiple stars'''

        # extract data from data_extraction
        data = Data()
        data.extract_from_query(query)
        df= data.data

        # align wavelengths with wavelength_alignment
        object_ids= list(df["specObjID"][0:5])
        min_val= 3.7
        max_val= 3.8
        num_points = 500
        aligned_x, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
        
        # compute derivative of aligned data
        aligned_deriv = DataAugmentor.compute_derivative(aligned_y, order=1)

        assert aligned_y.shape == aligned_deriv.shape

    def test_augment_data_derivative_unaligned_spectra(self):
        '''Compute derivatives of unaligned spectra for one star'''

        # extract data from data_extraction
        data = Data()

        # unaligned spectra
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]
        unaligned_spectra = pd.DataFrame(new_array)     

        # preprocess unaligned spectra
        unaligned_spectra = Preprocessing.correct_redshift(redshift = data.data['redshift'][0],data = unaligned_spectra)
        unaligned_spectra_std = Preprocessing.normalize(unaligned_spectra)

        unaligned_deriv = DataAugmentor.compute_derivative(unaligned_spectra_std, order=1)

        assert unaligned_spectra.shape == unaligned_deriv.shape
