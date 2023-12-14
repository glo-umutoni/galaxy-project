'''Test the integration of the interactive_visualization module into data pipeline.'''
import sys
sys.path.append("app/")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.widgets
import matplotlib.pyplot as plt
from data_extraction import Data
from wavelength_alignment import WavelengthAlignment
from preprocessing import Preprocessing
from interactive_visualization import InteractiveVisualization
matplotlib.use('Agg')

query="SELECT TOP 10 * FROM SpecObj"

def perform_alignment(object_ids):
    '''Performs wavelength alignment for desired spectral objects and returns as a pandas DataFrame.'''
    min_val= 3.7
    max_val= 3.8
    num_points = 5000
    aligned_x, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
    
    df_aligned = pd.DataFrame()
    df_aligned["loglam"] = aligned_x
    for i,flux in enumerate(aligned_y):
        df_aligned[f"flux_{i}"] = aligned_y[i]
    return df_aligned

class TestIntegrationInteractiveVisualization:
    def test_plot_raw(self, monkeypatch):
        '''Test interactive plotting of spectral data directly from the Data class, without preprocessing.'''
        monkeypatch.setattr(plt, 'show', lambda: None)
        data = Data()
        data.extract_from_query(query)
        spectrum = data.get_spectra_from_data()
        df = pd.DataFrame(spectrum[0][1].data)
        
        output = InteractiveVisualization.plot(df, y_column="flux")
        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)
    
    def test_plot_aligned(self, monkeypatch):
        '''Test interactive plotting of spectral data after wavelength alignment.'''
        monkeypatch.setattr(plt, 'show', lambda: None)
        data = Data()
        data.extract_from_query(query)

        # perform wavelength alignment
        object_ids= list(data.data["specObjID"][0:5])
        df_aligned = perform_alignment(object_ids=object_ids)

        # test plotting aligned spectra
        fig, ax = plt.subplots()
        output = InteractiveVisualization.plot(df_aligned, figax=(fig, ax), y_column="flux_0")
        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)

    def test_plot_processed(self, monkeypatch):
        '''Test interactive plotting of spectral data after alignment and preprocessing.'''
        monkeypatch.setattr(plt, 'show', lambda: None)
        data = Data()
        data.extract_from_query(query)

        # perform wavelength alignment
        object_ids= list(data.data["specObjID"][0:5])
        df_aligned = perform_alignment(object_ids=object_ids)

        # perform preprocessing functions on spectral data
        df_processed = Preprocessing.normalize(data=df_aligned)
        df_processed = Preprocessing.remove_outliers(data=df_processed)
        df_processed = Preprocessing.correct_redshift(data=df_processed, redshift=10)

        # test plotting aligned and preprocessed spectra
        fig, ax = plt.subplots()
        output = InteractiveVisualization.plot(df_processed, figax=(fig, ax), y_column="flux_0")
        # check that function returns a matplotlib.widgets.SpanSelector object
        assert isinstance(output, matplotlib.widgets.SpanSelector)