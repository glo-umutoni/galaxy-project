'''Module used to provide interactive matplotlib-based interface. User can select plot regions and quantify total flux.'''
from typing import Tuple
from astroquery.sdss import SDSS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.widgets import SpanSelector
from scipy.signal import savgol_filter
from visualization import Visualization

class InteractiveVisualization:
    '''Provides interactive visualization functionality given spectra data.'''

    @staticmethod
    def calc_line_area(x:list[float], y:list[float]):
        '''Returns the discrete approximation of area under a line with values y evaluated at points x
        
        Parameters
        ----------
        x : list[float]
            List of points at which line is evaluated
        
        y : list[float]
            The line evaluated at discrete points x 

        Returns
        ------
        area : float
            The discrete approximation of y integrated along x. Calls numpy's trapz() function, which
            approximates an integral using the composite trapezoidal rule.
        '''
        area = np.trapz(y, x)
        return area
    
    @staticmethod
    def plot(data, y_column:str, order:int=3, figax:Tuple[matplotlib.figure.Figure ,matplotlib.axes.Axes] =None, **kwargs) -> matplotlib.figure.Figure:
        '''
        Creates interactive plot where user can select region and quantify the flux of spectral line.
        Calls the Visualization class plot() function before adding interactivity.

        Parameters
        ----------
        data : pandas DataFrame
            pandas DataFrame object containing spectral data for visualization. 
            Must contain columns "loglam" and y_column.
        
        y_column : str
            column of the dataframe to plot as y.
        figax : Tuple[matplotlib.figure.Figure ,matplotlib.axes.Axes]
            fig, ax to use if given
        order : int, optional
            order of the polynomial used to fit the spectral data when denoising. Default is 3.
        **kwargs : arguments to be put in the plt.plot function
        Returns
        ------
        None

        Raises
        ------
        TypeError
            Raised if data is not a pandas DataFrame object
            Raised if y_column is not a string
        ValueError
            Raised if data does not contain the necessary columns
        '''
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Spectra input must be a pandas DataFrame.")
        
        if type(y_column)!=str:
            raise TypeError("y_column should be a string.")
        
        if not all([key in data.columns for key in ["loglam", y_column]]):
            raise ValueError("Spectra DataFrame must contain columns 'loglam' and y_column.")

        if figax is None:
            fig,ax=plt.subplots()
        else:
            fig,ax=figax

        # first create base plot without interactivity
        fig = Visualization.plot(data, y_column, order, (fig,ax), **kwargs)
        
        # get inferred continuum
        wavelengths = 10**data["loglam"] # undo log
        flux = data[y_column]
        # if no window size is specified, use the size of data
        if not "window_size" in kwargs: 
            window_size = len(flux)
        else:
            window_size = kwargs["window_size" ]
        denoised_flux = savgol_filter(flux, window_size, polyorder=order)

        # text indicating total flux; initially N/A until user selects region
        default_flux = "N/A"
        text = ax.text(min(wavelengths), max(flux), f"Total flux: {default_flux}", ha='left', fontweight="semibold")

        def on_select(xmin, xmax):
            '''Update total flux displayed on plot when user selects region'''
            i_min, i_max = np.searchsorted(wavelengths, (xmin, xmax))
            i_max = min(len(wavelengths) - 1, i_max)

            region_x = wavelengths[i_min:i_max]
            region_y = denoised_flux[i_min:i_max]

            if len(region_x) >= 2:
                total_flux = InteractiveVisualization.calc_line_area(region_x,region_y)
                total_flux = total_flux * (10**-9) # convert from maggies to nanomaggies
                text.set_text(f"Total flux: {total_flux:.6f} maggies")
                plt.draw()

        # allows user to interactively select a region on plot
        span = SpanSelector(
            ax,
            on_select,
            "horizontal",
            useblit=True,
            props=dict(alpha=0.5, facecolor="turquoise"),
            interactive=True,
            drag_from_anywhere=True
        )

        # fix the location of the legend
        legend = ax.get_legend()
        legend.set_loc("upper right")

        plt.show()

        return span