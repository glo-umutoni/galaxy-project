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
matplotlib.use('Agg')

# sample query and spectra data for testing purposes
query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)

class TestVisualization:
    '''Test the Visualization class from the visualization module'''

    def test_plot_bad_input(self):
        '''Test plot function when inputs are invalid'''
        # check that incorrect input type raises exception
        with pytest.raises(TypeError): 
            Visualization.plot([], y_column="flux")

        # check that correct input type with missing y_columns raises exception
        with pytest.raises(ValueError):
            Visualization.plot(pd.DataFrame(), y_column="flux")
        
        with pytest.raises(ValueError):
            Visualization.plot(data=data, y_column="Notflux")
        
        with pytest.raises(TypeError):
            Visualization.plot(data=data, y_column=36)

    def test_plot(self):
        '''Test plot function when input is valid'''
        # check that plot function with correct input runs
        output = Visualization.plot(data, y_column="flux")

        # check that function returns a matplotlib figure
        assert isinstance(output, matplotlib.figure.Figure)

        # check that custom keyword arguments work
        output = Visualization.plot(data, y_column="flux", order=3)

        # check that function returns a matplotlib figure
        isinstance(output, matplotlib.figure.Figure)

        fig, ax = plt.subplots()
        output = Visualization.plot(data, y_column="flux", figax = (fig, ax), order=3)
        isinstance(output, matplotlib.figure.Figure)