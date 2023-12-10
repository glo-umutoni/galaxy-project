'''Module used to extract and/or combine astronomical data from Sloan Digital Sky Survey.'''
import sys
sys.path.append("/app/")
from preprocessing import Preprocessing

class WavelengthAlignment:
    '''Retrieves and aligns wavelengths for specified objects
    Parameters
    ----------
    object_ids: a list of object id integers to align
    min_val: lower range boung=d of the log wavelength
    max_val: upper range bound of the log wavelength
    num_points: number of interpolated points to return

    Returns:
    --------
     aligned_x: array of linearly spaced log wavelengths
     aligned_y: list of arrays of interpolated flux based on the user selected range and number of points
    Raises:
    -------
    '''

    @staticmethod
    def align(object_ids: list, min_val: (int,float), max_val: (int,float), num_points:int):
        if type(num_points) != int:
            raise ValueError("Number of points to interpolate needs to be an integer")
        if type(object_ids) != list:
            raise ValueError("Object ids to interpolate must be a list")
        if min_val == 0:
            raise ValueError("Range lower bound cannot be empty")
        if max_val == 0:
            raise ValueError("Range upper bound cannot be empty")

        aligned_y=[]
        for object in (object_ids):
            x, y = Preprocessing.interpolate(object_id=object, min_val=min_val, max_val=max_val, num_points=num_points)
            aligned_x= x
            aligned_y.append(y)
        return aligned_x,aligned_y