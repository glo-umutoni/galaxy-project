'''Tests for integration of classification module'''

import numpy as np
import sys
import pandas as pd
sys.path.append("app/")
from data_extraction import Data
from preprocessing import Preprocessing
from classification import Classifier
from wavelength_alignment import WavelengthAlignment
query="SELECT TOP 10 * FROM SpecObj"

class TestIntegrationClassifier:
    '''Classify after extracting/modifying data with data_extraction, 
    preprocessing, and wavelength_alignment'''

    def test_predict_preprocessed_aligned_data(self):
        # data_extraction --> wavelength_alignment --> preprocessing --> classification
        data = Data()
        data.extract_from_query(query)
        data.get_spectra_from_data()

        # align spectra 
        min_val = 1
        max_val = 3
        object_ids = data.data['objid']
        num_points = 5
        _, aligned_spectra = WavelengthAlignment.align(object_ids, min_val, max_val, num_points)

        # preprocess data
        std_spectra = Preprocessing.normalize(data=aligned_spectra)
        metadata = data.data.drop(columns = ['class', 'objid','fiberid'])
        std_metadata = Preprocessing.normalize(data=metadata)

        # combine metadata and spectra
        X = pd.concat([std_metadata,std_spectra.T],axis=1)
        y = data.data['class']

        # perform object prediction
        classifier = Classifier('LogisticRegression')
        classifier.fit(X,y)
        y_pred = classifier.predict(X)
        assert y_pred.shape == y.shape
        assert classifier.predict_proba(X).shape == (len(y),3)
        assert 0 <= classifier.score(X,y) <= 1
        assert classifier.confusion_matrix(y, y_pred).shape == (3,3)