'''Module used to provide matplotlib-based interface for spectral visualization.'''
from typing import Tuple
from astroquery.sdss import SDSS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

class Visualization:
    '''Provides visualization functionality given spectra data.'''

    @staticmethod
    def plot(data, y_column:str, order:int=2, figax:Tuple[matplotlib.figure.Figure ,matplotlib.axes.Axes] =None, **kwargs) -> matplotlib.figure.Figure:
        '''
        Plots spectra data with overlay of inferred continuuum

        Parameters
        ----------
        data : pandas DataFrame
            pandas DataFrame object containing spectral data for visualization. 
            Must contain columns "loglam" and "flux".
        
        y_column : str
            column of the dataframe to plot as y.

        figax : Tuple[matplotlib.figure.Figure ,matplotlib.axes.Axes]
            fig, ax to use if given

        order : int, optional
            order of the polynomial used to fit the spectral data when denoising. Default is 3.

        **kwargs : arguments to be put in the plt.plot function

        Returns
        ------
        fig : matplotlib.figure.Figure
            plot of spectral data

        OR

        None : if ax is passed

        Raises
        ------
        TypeError
            Raised if spectra is not a pandas DataFrame object
            Raised if y_column is not a string
        ValueError
            Raised if spectra does not contain the necessary columns
        '''
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Spectra input must be a pandas DataFrame.")
        
        if type(y_column)!=str:
            raise TypeError("y_column should be a string.")
        
        if not all([key in data.columns for key in ["loglam", y_column]]):
            raise ValueError("Spectra DataFrame must contain columns 'loglam' and y_column.")

        
        wavelengths = 10**data["loglam"] # undo log
        flux = data[y_column]

        if figax is None:
            fig,ax=plt.subplots()
        else:
            fig,ax=figax
        
        # if no window size is specified, use the size of data
        if not "window_size" in kwargs: 
            window_size = len(flux)
        else:
            window_size = kwargs["window_size" ]

        # get inferred continuum
        denoised_flux = savgol_filter(flux, window_size, polyorder=order)
        
        # plot spectral data and inferred continuum
        ax.plot(wavelengths, flux, color="dodgerblue", linewidth=0.25, **kwargs)
        ax.plot(wavelengths, denoised_flux, color="darkviolet", linewidth=1.5, **kwargs)
        custom_lines = [Line2D([0],[0], color="dodgerblue", lw=4, label="Flux"),
                        Line2D([0],[0], color="darkviolet", lw=4, label="Inferred continuum")]
        ax.legend(handles=custom_lines, fancybox=True)
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel("Flux (nanomaggies)")

        return fig