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
            DataAugmentor.compute_derivative([[10]], [2])

        with pytest.raises(ValueError):
            DataAugmentor.compute_derivative(np.random.rand(1, 2), [None])

        with pytest.raises(ValueError):
            DataAugmentor.compute_derivative([10], [2])

    def test_compute_derivative_return_value(self):
        '''Check that a 2-D array is returned'''
        x = pd.DataFrame({"flux":np.random.rand(10)})
        print("flux" not in x)
        augmented_data = DataAugmentor.compute_derivative(data=x, derivative_order=[2])
        assert augmented_data.shape[0] == 10
        assert "flux_2" in augmented_data


    def test_compute_derivative_correct_values(self):
        '''Check that correct values are returned'''
        x = pd.DataFrame({"flux":np.random.rand(10)})
        augmented_data = DataAugmentor.compute_derivative(data=x, derivative_order=[2])
        differint_aug_data = GL(f_name=x["flux"].array, alpha=2, num_points=len(x["flux"]))
        assert np.array_equal(augmented_data["flux_2"], differint_aug_data)