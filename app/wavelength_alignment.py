'''Module used to extract and/or combine astronomical data from Sloan Digital Sky Survey.'''

from astroquery.sdss import SDSS
import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class WavelengthAlignment:
    '''Retrieves and stores SDSS data within class.'''

    def __init__(self):
        self.interpolated_x = None
        self.interpolated_y = []

    def interpolate(self, object_id, min_val, max_val, num_points:int):
        object_id_str = str(object_id)
        query = rf"SELECT * FROM SpecObj where specObjID in ({object_id_str})"
        query_result = SDSS.query_sql(query)
        spectra = SDSS.get_spectra(matches=query_result)
        spectra_data = pd.DataFrame(spectra[0][1].data)
        loglam = np.array(spectra_data['loglam'], dtype=float)
        flux = np.array(spectra_data['flux'], dtype=float)
        interp_function = interp1d(x=loglam, y=flux, kind='linear', fill_value='extrapolate')
        linear_wavelengths = np.linspace(min_val, max_val, num_points)
        interpolated_flux = interp_function(linear_wavelengths)
        return linear_wavelengths, interpolated_flux

    def align(self, object_ids: list, min_val, max_val, num_points:int):
        for object in object_ids:
            inter_x, inter_y = self.interpolate(object_id=object, min_val=min_val, max_val=max_val, num_points=num_points)
            self.interpolated_x = inter_x
            self.interpolated_y.append(inter_y)