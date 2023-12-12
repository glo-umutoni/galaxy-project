import os
import shutil
import pytest
import numpy as np
import sys
sys.path.append("../app/")
from wavelength_alignment import WavelengthAlignment

class TestWavelengthAlignment:
    '''Test the WavelengthAlignment class from the wavelength_alignment module'''

    def test_interpolate_handle_bad_inputs(self):
        '''Check that bad inputs raise the correct errors'''
        incorrect_min_val, incorrect_max_val = 0, 0
        incorrect_object_ids = (299489677444933632, 299489677444933632)
        incorrect_pts = 312.2

        with pytest.raises(ValueError):
            # incorrect entries for object ids
            WavelengthAlignment.align(object_ids=incorrect_object_ids, min_val=3.5, max_val=3.7, num_points=500)
        
        with pytest.raises(ValueError):
            # incorrect minimum value
            WavelengthAlignment.align(object_ids=[299489677444933632, 299489677444933632], min_val=incorrect_min_val
                                      ,max_val=3.7, num_points=500)

        with pytest.raises(ValueError):
            # incorrect maximum value
            WavelengthAlignment.align(object_ids=[299489677444933632, 299489677444933632], min_val=3.5
                                      , max_val=incorrect_max_val, num_points=500)
        
        with pytest.raises(ValueError):
            # incorrect object ids
            WavelengthAlignment.align(object_ids=[299489677444933632, 299489677444933632], min_val=3.5
                                      , max_val=3.7, num_points=incorrect_pts)

    def test_align_correct_values(self):
        '''check handling of correct inputs'''
        min_val = 3.2
        max_val = 3.7
        num_points = 500
        object_ids = [299489677444933632, 299489677444933632]
        aligned_x, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val,
                                                         num_points=num_points)
        assert type(aligned_x) == np.ndarray
        assert type(aligned_y) == list
        assert len(aligned_x) == num_points
        assert len(aligned_y) == len(object_ids)
        assert min(aligned_x) == min_val
        assert max(aligned_x) == max_val