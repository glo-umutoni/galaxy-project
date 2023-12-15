'''Tests for integration of classification module'''

import numpy as np
import sys
import pandas as pd
sys.path.append("app/patapoufsinthestars")
from data_extraction import Data
from preprocessing import Preprocessing
from classification import Classifier
from wavelength_alignment import WavelengthAlignment
from sklearn.preprocessing import MultiLabelBinarizer
query="SELECT TOP 20 SpecObjID, ra,dec,z, run2d, class FROM SpecObj"

class TestIntegrationClassifier:
    '''
    Classify after extracting/modifying data with data_extraction, 
    preprocessing, and wavelength_alignment
    
    Tests: data_extraction --> wavelength_alignment --> preprocessing --> classification
    '''

    def test_predict_preprocessed_aligned_data(self):
        data = Data()
        data.extract_from_query(query)
        
        # align spectra 
        min_val = 1
        max_val = 3
        object_ids = list(data.data['SpecObjID'])
        num_points = 5
        aligned_spectra = WavelengthAlignment.align(object_ids, min_val, max_val, num_points)

        # drop IDs
        metadata = data.data.drop(columns = ['SpecObjID'])

        X, y = Classifier.data_for_classifier(metadata, merge_data=aligned_spectra)

        # normalize data
        X = Preprocessing.normalize(data=X)
    
        # perform object prediction
        classifier = Classifier('LogisticRegression')
        classifier.fit(X,y)
        y_pred = classifier.predict(X)
        assert y_pred.shape == y.shape
        assert classifier.predict_proba(X).shape == (len(y),3)
        assert 0 <= classifier.score(X,y) <= 1
        assert classifier.confusion_matrix(y, y_pred).shape == (3,3)