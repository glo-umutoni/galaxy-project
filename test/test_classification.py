import pytest
import numpy as np
import sys
sys.path.append("app/patapoufsinthestars")
from classification import Classifier

class TestInit:
    '''Test init method in Classifier'''

    def test_init_bad_inputs(self):
        '''Bad inputs are properly handled'''
        with pytest.raises(ValueError):
            Classifier("RandomModel")

class TestFit:
    '''Test fit method of Classifier'''

    def test_fit_bad_inputs(self):
        '''Bad inputs are properly handled'''
        return_values = ["GALAXY", "QSO", "STAR"]
        classifier = Classifier("KNeighborsClassifier")
        x_1d = np.random.rand(3)
        x_2d = np.random.rand(3,2)
        y_correct = return_values
        y_large = return_values + ['STAR']
        y_wrong = np.random.rand(3)
        with pytest.raises(ValueError):
            classifier.fit([],[])
        with pytest.raises(ValueError):
            # x should be a 2d array
            classifier.fit(x_1d, y_correct)
        with pytest.raises(ValueError):
            # x.shape[0] and len(y) should be the same
            classifier.fit(x_2d, y_large)
        with pytest.raises(ValueError):
            # y should contain "GALAXY", "QSO", "STAR"
            classifier.fit(x_2d, y_wrong)

    def test_fit_return_none(self):
        '''Tests that fit method returns None'''
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"]
        classifier = Classifier("KNeighborsClassifier")
        assert classifier.fit(x,y) is None


class TestPredict:
    '''Test predict method of Classifier'''

    def test_predict_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier("LogisticRegression")
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"]
        classifier.fit(x,y)
        x_new = np.random.rand(3,3)
        with pytest.raises(ValueError):
            # x_new does not match column dims of fitted x
            y_pred = classifier.predict(x_new)
        with pytest.raises(ValueError):
            # empty list input to predict
            y_pred = classifier.predict([])

    def test_predict_return_values(self):
        '''1d array of correct length is returned'''
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"] 
        x_new = np.random.rand(4,2)
        classifier = Classifier("RandomForestClassifier")
        classifier.fit(x,y)
        assert classifier.predict(x_new).shape == (x_new.shape[0],)

    def test_predict_correct_values(self):
        '''Tests that the correct predictions are returned'''
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"] 
        x_new = np.random.rand(4,2)
        classifier = Classifier("LogisticRegression")
        classifier.fit(x,y)
        # check that predictions are one of "GALAXY", "QSO", "STAR"
        assert set(classifier.predict(x_new)).issubset(y)


class TestPredictProba:
    '''Test predict_proba method of Classifier'''

    def test_predict_proba_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier("LogisticRegression")
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"] 
        classifier.fit(x,y)
        x_new = np.random.rand(3,3)
        with pytest.raises(ValueError):
            # x_new does not match column dims of fitted x
            classifier.predict_proba(x_new)
        with pytest.raises(ValueError):
            # empty list input to predict
            classifier.predict_proba([])

    def test_predict_proba_return_values(self):
        '''2d array is returned with 3 columns'''
        classifier = Classifier("LogisticRegression")
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"] 
        x_new = np.random.rand(4,2)
        classifier.fit(x,y)
        # returned array has correct shape
        assert classifier.predict_proba(x_new).shape == (x_new.shape[0],3)
        # probabilities sum to 1 across rows
        assert np.all(np.allclose(classifier.predict_proba(x_new).sum(axis=1),np.ones(x_new.shape[0]),atol=0.01))


class TestScore:
    '''Test score method of Classifier'''

    def test_score_bad_inputs(self):
        '''Bad inputs are properly handled'''
        return_values = ["GALAXY", "QSO", "STAR"]
        classifier = Classifier("LogisticRegression")
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"] 
        classifier.fit(x,y)
        x_new = np.random.rand(4,2)
        y_new = np.random.choice(return_values, 3)
        with pytest.raises(ValueError):
            # x_new and y have different lengths
            classifier.score(x_new, y)
        with pytest.raises(ValueError):
            # empty lists input to score
            classifier.score([],[])
        with pytest.raises(ValueError):
            # wrong dims
            classifier.score(y, y_new)

    def test_score_return_value(self):
        '''Proper score is returned'''
        x = np.random.rand(3,2)
        y = ["GALAXY", "QSO", "STAR"]
        classifier = Classifier("LogisticRegression")
        classifier.fit(x,y)
        # should return a float:
        assert isinstance(classifier.score(x,y), float)
        # should be accuracy (btw 0 and 1):
        assert 0 <= classifier.score(x,y) <= 1


class TestConfusionMatrix:
    '''Test confusion_matrix method of Classifier'''

    def test_confusion_matrix_bad_inputs(self):
        '''Bad inputs are properly handled'''
        classifier = Classifier("LogisticRegression")
        with pytest.raises(ValueError):
            classifier.confusion_matrix(np.random.rand(3), np.random.rand(4))

    def test_confusion_matrix_return_values(self):
        '''3x3 matrix returned'''
        y_true = ["GALAXY", "QSO", "STAR", "QSO", "GALAXY"]
        y_pred = ["GALAXY", "QSO", "STAR", "STAR", "QSO"]
        classifier = Classifier("LogisticRegression")
        assert classifier.confusion_matrix(y_true, y_pred).shape == (3,3)
    