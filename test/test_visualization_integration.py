'''Test the integration of the visualization module into data pipeline.'''
import sys
sys.path.append("app/patapoufsinthestars")
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from data_extraction import Data
from wavelength_alignment import WavelengthAlignment
from preprocessing import Preprocessing
from visualization import Visualization
matplotlib.use('Agg')

query="SELECT TOP 10 * FROM SpecObj"

def perform_alignment(object_ids):
    '''Performs wavelength alignment for desired spectral objects and returns as a pandas DataFrame.'''
    min_val= 3.7
    max_val= 3.8
    num_points = 5000
    df_aligned = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
    
    return df_aligned

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
        data.extract_from_query(query)

        # perform wavelength alignment
        object_ids= list(data.data["specObjID"][0:5])
        df_aligned = perform_alignment(object_ids=object_ids)

        # test plotting multiple aligned spectra
        fig, axs = plt.subplots(ncols=2)
        fig = Visualization.plot(df_aligned, figax=(fig, axs[0]), y_column=f"flux_{object_ids[0]}")
        fig = Visualization.plot(df_aligned, figax=(fig, axs[1]), y_column=f"flux_{object_ids[1]}")
        # check that return value is of correct type
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_plot_processed(self):
        '''Test plotting spectral data after alignment and preprocessing.'''
        data = Data()
        data.extract_from_query(query)

        # perform wavelength alignment
        object_ids= list(data.data["specObjID"][0:5])
        df_aligned = perform_alignment(object_ids=object_ids)

        # perform preprocessing functions on spectral data
        df_processed = Preprocessing.normalize(data=df_aligned)
        df_processed = Preprocessing.remove_outliers(data=df_processed)
        df_processed = Preprocessing.correct_redshift(data=df_processed, redshift=10)

        # test plotting multiple aligned and preprocessed spectra
        fig, axs = plt.subplots(ncols=2)
        fig = Visualization.plot(df_processed, figax=(fig, axs[0]), y_column=f"flux_{object_ids[0]}")
        fig = Visualization.plot(df_processed, figax=(fig, axs[1]), y_column=f"flux_{object_ids[1]}")
        # check that return value is of correct type
        assert isinstance(fig, matplotlib.figure.Figure)