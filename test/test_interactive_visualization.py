import sys
from astroquery.sdss import SDSS
import pandas as pd
import numpy as np
import matplotlib.figure
import matplotlib.widgets
import matplotlib.pyplot as plt
import pytest
sys.path.append("app/patapoufsinthestars")
from interactive_visualization import InteractiveVisualization
matplotlib.use('Agg')

# sample query and spectra data for testing purposes
query = "SELECT TOP 10 * FROM SpecObj"
query_result = SDSS.query_sql(query)
data = pd.DataFrame(SDSS.get_spectra(matches=query_result)[0][1].data)

class TestInteractiveVisualization:
    '''Test the InteractiveVisualization class from the interactive_visualization module'''

    def test_calc_line_area(self):
        '''Test the calc_line_area functionn'''
        x = 10**np.linspace(0,1,100)
        y = np.random.rand(100)
        line_area = InteractiveVisualization.calc_line_area(x,y)
        assert isinstance(line_area, float)

    def test_plot_bad_input(self, monkeypatch):
        '''Test plot function when inputs are invalid'''
        monkeypatch.setattr(plt, 'show', lambda: None)
        # check that incorrect input type raises exception
        with pytest.raises(TypeError): 
            InteractiveVisualization.plot([], y_column="flux")

        # check that correct input type with missing y_columns raises exception
        with pytest.raises(ValueError):
            InteractiveVisualization.plot(pd.DataFrame(), y_column="flux")
        
        with pytest.raises(ValueError):
            InteractiveVisualization.plot(data=data, y_column="Notflux")
        
        with pytest.raises(TypeError):
            InteractiveVisualization.plot(data=data, y_column=36)

    def test_plot(self, monkeypatch):
        '''Test plot function when input is valid'''
        monkeypatch.setattr(plt, 'show', lambda: None)
        # check that plot function with correct input runs
        output = InteractiveVisualization.plot(data, y_column="flux")

        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)

        # check that custom keyword arguments work
        output = InteractiveVisualization.plot(data, y_column="flux", order=3)

        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)

        fig, ax = plt.subplots()
        output = InteractiveVisualization.plot(data, y_column="flux", figax = (fig, ax), order=3)
        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)