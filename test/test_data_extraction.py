import os
import pandas as pd
import shutil
import numpy as np
from astroquery.sdss import SDSS
import pytest
import sys
sys.path.append("app/")
from data_extraction import Data

test_dataset_path = "test/data/test_dataset.csv"
test_sandbox_directory_path = "test/test_env"

@pytest.fixture(autouse=True)
def clean_up():
    os.makedirs(test_sandbox_directory_path, exist_ok=True)
    yield
    shutil.rmtree(test_sandbox_directory_path)

class TestData:
    '''Test the Data class from the data_extraction module'''

    def test_init(self):
        data = Data()
        assert type(data.data)==pd.DataFrame


    def test_extraction_from_query(self):
        query = "SELECT TOP 10 p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, \
            p.run, p.rerun, p.camcol, p.field, s.specobjid, s.class, s.z\
             as redshift, s.plate, s.mjd, s.fiberid FROM PhotoObj AS p JOIN\
             SpecObj AS s ON s.bestobjid = p.objid WHERE p.u BETWEEN 0 AND \
            19.6 AND g BETWEEN 0 AND 20"
        data = Data()
        data.extract_from_query(query)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 18)

    def test_extract_from_constraints(self):
        constrains = {"columns":["objid", "ra", "u", "g"],
                      "database" : ["PhotoObj"],
                      "constrain" : [("u", [0, 19.6]), ("g", [None, 20]), ("g", [0, None])]
                    }

        data = Data()
        data.extract_from_constraints(constrains)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (500000, 4)

    def test_extract_from_file(self):
        file_name = test_dataset_path 
        data = Data()
        data.extract_from_file(file_name)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 18)
    
    def test_get_spectra_from_obj_id(self):
        query="SELECT TOP 10 * FROM SpecObj"
        query_result = SDSS.query_sql(query)
        df = query_result.to_pandas()
        obj_id= str(df["specObjID"][0])
        spectra = SDSS.get_spectra(matches=query_result)
        spectra1 = Data.get_spectra_from_obj_id(obj_id=obj_id)
        assert np.array_equal(spectra[0][1].data, spectra1[0][1].data)

    def test_get_spectra_from_data(self):
        query="SELECT TOP 10 * FROM SpecObj"
        data = Data()
        data.extract_from_query(query)
        spectra1 = data.get_spectra_from_data()
        query_result = SDSS.query_sql(query)
        spectra = SDSS.get_spectra(matches=query_result)
        assert np.array_equal(spectra[0][1].data, spectra1[0][1].data)

    def test_write_file(self):
        file_name = test_dataset_path 
        output_file_path = os.path.join(test_sandbox_directory_path, "output.csv")
        data = Data()
        data.extract_from_file(file_name)
        data.write_file(output_file_path)
        assert os.path.isfile(output_file_path)

    def test_concat(self):
        file_name = test_dataset_path 
        data = Data()
        data.extract_from_file(file_name)
        data2 = Data()
        data2.extract_from_file(file_name)
        data.concat(data2, axis=0)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (20, 18)
        
    def test_merge(self):
        file_name = test_dataset_path 
        data = Data()
        data.extract_from_file(file_name)
        data2 = Data()
        data2.extract_from_file(file_name)
        data.merge(data2, on_column="objid")
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 35)