import os
import pandas as pd
import shutil
import pytest
import sys
sys.path.append("../app")
from data_extraction import Data

@pytest.fixture(autouse=True)
def clean_up():
    os.makedirs("test_env", exist_ok=True)
    yield
    shutil.rmtree("test_env")

class TestData:

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
        file_name = "data/test_dataset.csv"
        data = Data()
        data.extract_from_file(file_name)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 18)

    def test_write_file(self):
        file_name = "data/test_dataset.csv"
        output_file = "test_env/output.csv"
        data = Data()
        data.extract_from_file(file_name)
        data.write_file(output_file)
        assert os.path.isfile(output_file)

    def test_concat(self):
        file_name = "data/test_dataset.csv"
        data = Data()
        data.extract_from_file(file_name)
        data2 = Data()
        data2.extract_from_file(file_name)
        data.concat(data2, axis=0)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (20, 18)
        
    def test_merge(self):
        file_name = "data/test_dataset.csv"
        data = Data()
        data.extract_from_file(file_name)
        data2 = Data()
        data2.extract_from_file(file_name)
        data.merge(data2, on_column="objid")
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 35)