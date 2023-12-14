'''Module used to extract and/or combine astronomical data from Sloan Digital Sky Survey.'''
import sys
import pandas as pd
import numpy as np
from astroquery.sdss import SDSS
sys.path.append("/app/")
from preprocessing import Preprocessing

class WavelengthAlignment:
    '''Class that aligns the wavelength of the objects.'''

    @staticmethod
    def align(object_ids: list, min_val: (int,float), max_val: (int,float), num_points:int):
        '''Retrieves and aligns wavelengths for specified objects
        Parameters
        ----------
        object_ids: a list of object id integers to align
        min_val: lower range bound of the log wavelength
        max_val: upper range bound of the log wavelength
        num_points: number of interpolated points to return

        Returns:
        --------
        pd.Dataframe obect with columns "loglam", "flux_{obj_id}" for all obj ids

        Raises
        --------
        ValueError
            Raised if num_points is not an integer
            Raised if object_ids is not a list
            Raised if min_val is empty
            Raised if max_val is empty
        '''
        if type(num_points) != int:
            raise ValueError("Number of points to interpolate needs to be an integer")
        if type(object_ids) != list:
            raise ValueError("Object ids to interpolate must be a list")
        if min_val == 0:
            raise ValueError("Range lower bound cannot be empty")
        if max_val == 0:
            raise ValueError("Range upper bound cannot be empty")

        aligned={}
        for object_id in (object_ids):
            object_id_str = str(object_id)
            query = rf"SELECT * FROM SpecObj where specObjID in ({object_id_str})"
            query_result = SDSS.query_sql(query)
            # Retrieve spectral data: spectra[0] stores all the files related to the spectra for the object of interest and spectrum is stored as a table in the second item of the list.
            spectra = SDSS.get_spectra(matches=query_result)
            spectra_data = pd.DataFrame(spectra[0][1].data)
            # Extract logarithm of wavelength and flux
            loglam = np.array(spectra_data['loglam'], dtype=float)
            flux = np.array(spectra_data['flux'], dtype=float)
            x, y = Preprocessing.interpolate(x=loglam, y=flux, x_lim=(min_val, max_val), bins=num_points)
            aligned[f"flux_{object_id_str}"]=y
        aligned["loglam"]=x
        return pd.DataFrame(aligned)