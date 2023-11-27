import os
import pandas as pd
import shutil
from ..app.data_extraction import Data

@pytest.fixture(autouse=True)
def clean_up():
    os.makedirs("test_env", exist_ok=True)
    yield
    shutil.rmtree("test_env")

class TestData():

    def test_init():
        data = Data()
        assert type(data.data)==pd.DataFrame


    def test_extraction_from_query():
        query = "SELECT TOP 10 p.objid,p.ra,p.dec,p.u,p.g,p.r,p.i,p.z, \
            p.run, p.rerun, p.camcol, p.field, s.specobjid, s.class, s.z\
             as redshift, s.plate, s.mjd, s.fiberid FROM PhotoObj AS p JOIN\
             SpecObj AS s ON s.bestobjid = p.objid WHERE p.u BETWEEN 0 AND \
            19.6 AND g BETWEEN 0 AND 20"
        data = Data().extract_from_query(query)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 18)

    def test_extract_from_constraints():
        constrains = {"columns":["objid", "ra", "u"],
                      "database" : ["PhotoObj"],
                      "constrain" : [("u", [0, 19.6])],
                    }

        data = Data().extract_from_constraints(constrains)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 3)

    def test_extract_from_file():
        file_name = "data/test_dataset.csv"
        data = Data().extract_from_file(file_name)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 18)

    def test_write_file():
        file_name = "data/test_dataset.csv"
        output_file = "test_env/output.csv"
        data = Data().extract_from_file(file_name)
        data.write_file(output_file)
        assert os.path.isfile(output_file)

    def test_concat():
        file_name = "data/test_dataset.csv"
        data = Data().extract_from_file(file_name)
        data.concat(Data().extract_from_file(file_name), axis=0)
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (20, 18)
        
    def test_merge():
        file_name = "data/test_dataset.csv"
        data = Data().extract_from_file(file_name)
        data.merge(Data().extract_from_file(file_name), on_column="objid")
        assert type(data.data)==pd.DataFrame
        assert data.data.shape== (10, 35)