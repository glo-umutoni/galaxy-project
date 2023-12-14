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
    def test_data_augmentation_derivative_for_multiple_aligned_obj(self):
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
        _, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
        
        # compute derivative of aligned data
        aligned_deriv = DataAugmentor.compute_derivative(np.array(aligned_y), derivative_order=[1])

        assert aligned_deriv.shape == (1, 5, 500)

    def test_data_augmentation_derivative_for_one_unaligned_obj_with_preprocessing(self):
        '''Compute derivative of unaligned spectra for one star'''

        # unaligned spectra
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]
        unaligned_spectra = pd.DataFrame(new_array)     

        # preprocess unaligned spectra
        unaligned_spectra = Preprocessing.correct_redshift(redshift = 10, data = unaligned_spectra)
        unaligned_spectra_std = Preprocessing.normalize(unaligned_spectra)
        unaligned_flux = unaligned_spectra_std["flux"].to_numpy().reshape(1,-1)
        obj_flux = np.array(unaligned_flux)
        unaligned_deriv = DataAugmentor.compute_derivative(obj_flux, derivative_order=[1])

        assert unaligned_deriv.shape == (1, 1, len(unaligned_spectra_std["flux"]))


    def test_data_augmentation_frac_derivative_for_multiple_aligned_obj(self):
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
        _, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
        
        # compute derivative of aligned data
        aligned_deriv = DataAugmentor.compute_derivative(np.array(aligned_y), derivative_order=[0.7])

        assert aligned_deriv.shape == (1, 5, 500)

    def test_data_augmentation_frac_derivative_for_single_unaligned_obj_with_preprocessing(self):
        '''Compute fractional derivative of unaligned spectra for one star'''

        # unaligned spectra
        rec_array = SDSS.get_spectra(matches=query_result)[0][1].data
        new_array = np.empty(rec_array.shape, dtype=new_dtype)
        for field in rec_array.dtype.names:
            new_array[field] = rec_array[field]
        unaligned_spectra = pd.DataFrame(new_array)     

        # preprocess unaligned spectra
        unaligned_spectra = Preprocessing.correct_redshift(redshift = 10, data = unaligned_spectra)
        unaligned_spectra_std = Preprocessing.normalize(unaligned_spectra)
        unaligned_flux = unaligned_spectra_std["flux"].to_numpy().reshape(1,-1)
        obj_flux = np.array(unaligned_flux)
        unaligned_deriv = DataAugmentor.compute_derivative(obj_flux, derivative_order=[0.7])

        assert unaligned_deriv.shape == (1, 1, len(unaligned_spectra_std["flux"]))

    def test_data_augmentation_frac_derivative_for_single_aligned_obj_for_multiple_derv(self):
        '''Compute derivatives of aligned spectra for one star'''
        min_val = 3.2
        max_val = 3.7
        num_points = 500
        object_ids = [299489677444933632]
        _, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)

        aug_data = DataAugmentor.compute_derivative(data=np.array(aligned_y), derivative_order=[0.5,0.7])
        assert aug_data.shape == (2, 1, 500)

    def test_data_augmentation_derivative_for_multiple_aligned_obj_for_multiple_derv(self):
        '''Compute derivatives of aligned spectra for multiple stars'''
        min_val = 3.2
        max_val = 3.7
        num_points = 500
        object_ids = [299489677444933632, 299489677444933632, 299490502078654464, 299490227200747520]
        _, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)

        aug_data = DataAugmentor.compute_derivative(data=np.array(aligned_y), derivative_order=[2,1])
        assert aug_data.shape == (2, 4, 500)