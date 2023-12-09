'''Module used to provide matplotlib-based interface for spectral visualization.'''

from astroquery.sdss import SDSS
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter

class Visualization:
    '''Provides visualization functionality given spectra data.'''

    def __init__(self):
        pass

    def plot(spectra, window_size=None, order=3):
        '''
        Plots spectra data with overlay of inferred continuuum

        Parameters
        ----------
        spectra : pandas DataFrame
            pandas DataFrame object containing spectral data for visualization. 
            Must contain columns "loglam" and "flux".

        window_size : int or None, optional
            length of the filter window used for denoising.
            Default is None, in which case window size gets set to the length of the input data.

        order : int, optional
            order of the polynomial used to fit the spectral data when denoising. Default is 3.

        Raises
        ------
        TypeError
            Raised if spectra is not a pandas DataFrame object
        ValueError
            Raised if spectra does not contain the necessary columns
        '''
        if not isinstance(spectra, pd.DataFrame):
            raise TypeError("Spectra input must be a pandas DataFrame.")
        
        if not all([key in spectra.columns for key in ["loglam", "flux"]]):
            raise ValueError("Spectra DataFrame must contain columns 'loglam' and 'flux'.")
        
        wavelengths = 10**spectra["loglam"] # undo log
        flux = spectra["flux"]
        
        # if no window size is specified, use the size of data
        if window_size is None: window_size = len(flux)

        # get inferred continuum
        denoised_flux = savgol_filter(flux, window_size, order)
        
        # plot spectral data and inferred continuum
        fig,ax=plt.subplots()
        ax.plot(wavelengths, flux, color="dodgerblue", linewidth=0.25)
        ax.plot(wavelengths, denoised_flux, color="darkviolet", linewidth=1.5)
        custom_lines = [Line2D([0],[0], color="dodgerblue", lw=4, label="Flux"),
                        Line2D([0],[0], color="darkviolet", lw=4, label="Inferred continuum")]
        ax.legend(handles=custom_lines, fancybox=True)
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel("Flux (nanomaggies)")

        return fig