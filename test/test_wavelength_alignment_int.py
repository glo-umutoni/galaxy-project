import numpy as np
import sys
sys.path.append("app/patapoufsinthestars")
from data_extraction import Data
from wavelength_alignment import WavelengthAlignment

query="SELECT TOP 10 * FROM SpecObj"

class TestIntegrationWavelengthAlignment:
    def test_align_data(self):
        data = Data()
        data.extract_from_query(query)
        df= data.data
        object_ids= list(df["specObjID"][0:5])
        min_val= 3.7
        max_val= 3.8
        num_points = 5000
        aligned_df = WavelengthAlignment.align(object_ids=object_ids, min_val=min_val, max_val=max_val, num_points=num_points)
        assert aligned_df.shape==(num_points, 6)
        assert "loglam" in aligned_df