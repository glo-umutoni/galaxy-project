import pandas as pd
import pytest
import sys
import numpy as np
from differint.differint import GL
sys.path.append("app/")
from data_augmentation import DataAugmentor

class TestDataAugmentor:
    '''Test the DataAugmentor class from the data_augmentation module'''

    def test_compute_derivative_handle_bad_inputs(self):
        '''Check that bad inputs are handled gracefully'''
        with pytest.raises(ValueError):
            DataAugmentor.compute_derivative(None, [2])

        with pytest.raises(ValueError):
            DataAugmentor.compute_derivative([[10]], [None])

    def test_compute_derivative_return_value(self):
        '''Check that a 2-D array is returned'''
        augmented_data = DataAugmentor.compute_derivative([[10, 20]], [2])
        assert augmented_data.shape[0] == 1
        assert augmented_data.shape[1] == 1
        assert augmented_data.shape[2] == 2

    def test_compute_derivative_correct_values(self):
        '''Check that correct values are returned'''
        random_flux = np.random.rand(10)
        augmented_data = DataAugmentor.compute_derivative([random_flux], [2])
        differint_aug_data = [GL(f_name=random_flux, alpha=2)]
        assert np.array_equal(augmented_data, differint_aug_data)