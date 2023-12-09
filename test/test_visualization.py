import os
import sys
from astroquery.sdss import SDSS
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot as plt
import shutil
import pytest
sys.path.append("app/")
from visualization import Visualization

# sample query and spectra data for testing purposes
query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)

class TestVisualization:
    '''Test the Visualization class from the visualization module'''

    def test_init(self):
        '''Test initialization'''
        vis = Visualization()

    def test_plot_bad_input(self):
        '''Test plot function when inputs are invalid'''
        # check that incorrect input type raises exception
        with pytest.raises(TypeError): 
            Visualization.plot([])

        # check that correct input type with missing columns raises exception
        with pytest.raises(ValueError):
            Visualization.plot(pd.DataFrame())

    def test_plot(self):
        '''Test plot function when input is valid'''
        # check that plot function with correct input runs
        output = Visualization.plot(data)

        # check that custom keyword arguments work
        Visualization.plot(data, window_size=11, order=2)

        # check that function returns a matplotlib figure
        assert isinstance(output, matplotlib.figure.Figure)