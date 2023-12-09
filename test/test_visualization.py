import os
import sys
from astroquery.sdss import SDSS
import pandas as pd
import matplotlib.pyplot as plt
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

    def test_plot(self):
        '''Test plot function'''
        # check that incorrect input raises exception
        with pytest.raises(TypeError): 
            Visualization.plot([])

        # check that plot function with correct input executes without error
        output = Visualization.plot(data)
        plt.show()

        # check for correct return value
        assert output is None