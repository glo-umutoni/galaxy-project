import os
import pandas as pd
import pytest
import numpy as np
import sys
sys.path.append("app/")
from data_extraction import Data
from classification import Classifier

# do I need to test the initialization?

class TestFit:
    '''Test fit method of Classifier'''

    def test_fit_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier()
        x_1d = np.random.rand(6)
        x_2d = np.random.rand(3,2)
        y_1d = np.random.rand(6)
        y_2d = np.random.rand(3,2)
        with pytest.raises(ValueError):
            # x should be a 2d array
            classifier.fit(x_1d,y_1d)
            # x.shape[0] and len(y) should be the same
            classifier.fit(x_2d, y_1d)
            # y should be 1d array
            classifier.fit(x_2d, y_2d)

    def test_fit_return_none(self):
        '''Tests that fit method returns None'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        assert classifier.fit(x,y) is None

    def test_fit_attribute_created(self):
        '''Tests whatever we want to have inside new attribute'''
        pass

class TestPredict:
    '''Test predict method of Classifier'''

    def test_predict_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        x_new = np.random.rand(3,3)
        with pytest.raises(ValueError):
            # x_new does not match column dims of fitted x
            classifier.predict(x_new)
            # empty list input to predict
            classifier.predict([])

    def test_predict_return_values(self):
        '''1d array of correct length is returned'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        x_new = np.random.rand(4,2)
        assert classifier.predict(x_new).shape == (x_new.shape[0],)

    def test_predict_correct_values(self):
        '''Tests that the correct predictions are returned'''
        # what values should this return? are we using converting to -1, 0, 1?
        pass

class TestPredictProba:
    '''Test predict_proba method of Classifier'''

    def test_predict_proba_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        x_new = np.random.rand(3,3)
        with pytest.raises(ValueError):
            # x_new does not match column dims of fitted x
            classifier.predict_proba(x_new)
            # empty list input to predict
            classifier.predict_proba([])

    def test_predict_proba_return_values(self):
        '''2d array is returned with 3 columns'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        x_new = np.random.rand(4,2)
        assert classifier.predict_proba(x_new).shape == (x_new.shape[0],3)
    
    def test_predict_proba_correct_values(self):
        '''Test that correct values are returned'''
        pass

class TestScore:
    '''Test score method of Classifier'''

    def test_score_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        x_new = np.random.rand(4,2)
        y_new = np.random.rand(3)
        with pytest.raises(ValueError):
            # x_new and y have different lengths
            classifier.score(x_new, y)
            # empty lists input to score
            classifier.score([],[])
            # wrong dims
            classifier.score(y, y_new)

    def test_score_return_value(self):
        '''Proper score is returned'''
        classifier = Classifier()
        x = np.random.rand(3,2)
        y = np.random.rand(3)
        classifier.fit(x,y)
        # should return a float:
        assert isinstance(classifier.score(x,y), float)
        # should be a probability:
        assert 0 <= classifier.score(x,y) <= classifier.score


    def test_score_correct_value(self):
        '''Returned score is correct'''
        pass