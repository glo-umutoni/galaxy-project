'''Test the integration of the visualization module into data pipeline.'''
import sys
sys.path.append("../app/")
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from data_extraction import Data
from wavelength_alignment import WavelengthAlignment
from preprocessing import Preprocessing
from visualization import Visualization

query="SELECT TOP 10 * FROM SpecObj"
df_aligned = None

class TestIntegrationVisualization:
    def test_plot_raw(self):
        '''Test plotting spectral data directly from the Data class, without preprocessing.'''
        data = Data()
        data.extract_from_query(query)
        spectrum = data.get_spectra_from_data()
        df = pd.DataFrame(spectrum[0][1].data)
        
        fig = Visualization.plot(df, y_column="flux")
        # check that return value is of correct type
        assert isinstance(fig, matplotlib.figure.Figure)
    
    def test_plot_aligned(self):
        '''Test plotting spectral data after wavelength alignment.'''
        data = Data()
        metadata = data.extract_from_query(query)

        # perform wavelength alignment
        object_ids= list(metadata["specObjID"][0:5])
        min_val= 3.7
        max_val= 3.8
        num_points = 5000
        aligned_x, aligned_y = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)

        # dataframe with aligned flux data
        global df_aligned
        df_aligned = pd.DataFrame()
        df_aligned["loglam"] = aligned_x
        for i,flux in enumerate(aligned_y):
            df_aligned[f"flux_{i}"] = aligned_y[i]

        # test plotting multiple aligned spectra
        fig, ax = plt.subplots()
        fig = Visualization.plot(df_aligned, ax=ax, y_column="flux_0")
        fig = Visualization.plot(df_aligned, ax=ax, y_column="flux_1")
        # check that return value is of correct type
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_processed(self):
        '''Test plotting spectral data after alignment and preprocessing.'''
        # perform preprocessing functions on spectral data
        df_processed = Preprocessing.normalize(data=df_aligned)
        df_processed = Preprocessing.remove_outliers(data=df_processed)
        df_processed = Preprocessing.correct_redshift(data=df_processed, redshift=10)

        # test plotting multiple aligned and preprocessed spectra
        fig, ax = plt.subplots()
        fig = Visualization.plot(df_processed, ax=ax, y_column="flux_0")
        fig = Visualization.plot(df_processed, ax=ax, y_column="flux_1")
        # check that return value is of correct type
        assert isinstance(fig, matplotlib.figure.Figure)