import os
import shutil
import pytest
import sys
sys.path.append("../app/")
from wavelength_alignment import WavelengthAlignment

class TestWavelengthAlignment:
    '''Test the WavelengthAlignment class from the wavelength_alignment module'''
    def test_init(self):
        wave_align= WavelengthAlignment()
        assert wave_align.interpolated_x is None
        assert type(wave_align.interpolated_y) == list

    def test_interpolate(self):
        object_id = 299489952322840576
        min_val, max_val = 3.7, 3.8
        num_points = 5000
        wv = WavelengthAlignment()
        linear_wavelengths, interpolated_flux=wv.interpolate(object_id=object_id, min_val=min_val, max_val=max_val, num_points=num_points)
        assert len(interpolated_flux) == 5000
        assert len(linear_wavelengths) ==5000

    def test_align(self):
        object_ids = [299489952322840576, 299489677444933632]
        min, max = 3.7, 3.8
        num_points = 5000
        wv = WavelengthAlignment()
        wv.align(object_ids=object_ids, min_val=min, max_val=max, num_points=num_points)
        inter_flux = wv.interpolated_y
        inter_loglam = wv.interpolated_x
        assert len(inter_flux) == 2
        assert len(inter_loglam) == 5000