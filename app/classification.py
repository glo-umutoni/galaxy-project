'''Module used to classify data from Sloan Digital Sky Survey as Star, Galaxy, or QSO.'''

from astroquery.sdss import SDSS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

class Classifier:
    '''Classification estimator adapted from sklearn model'''

    Models = {
            "KNeighborsClassifier":KNeighborsClassifier,
            "LogisticRegression":LogisticRegression,
            "RandomForestClassifier":RandomForestClassifier
        }

    def __init__(self, model_name, **kwargs):
        '''Initialize estimator object from sklearn

        Parameters
        ----------
        model_name : str
            Name of sklearn classifier to use for object identification.
            Must be one of 'KNeighborsClassifier', 'LogisticRegression',
            and 'RandomForestClassifier'.

        **kwargs : Parameters for specified classifier.  

        Example
        ----------
        If the base estimator desired is KNeighborsClassifier(n_neighbors=5),
        then use:
        >>> Classifier(model_name='KNeighborsClassifier', n_neighbors=5)
        '''
        
        if model_name not in {key for key in Models}:
            raise ValueError("Must be one of 'KNeighborsClassifier','LogisticRegression','RandomForestClassifier'")
        
        self.model = Models[model_name](**kwargs)

    def fit(self, x, y):
        '''Fit classifier.
        
        Parameters
        ----------
        x : array_like (2d)
            Spectral and/or metadata information for sky objects. 
            Should already be preprocessed.

        y : array_like (1d)
            Array containing the values "GALAXY", "QSO", and/or "STAR", 
            the true classes for the sky objects.  
        '''
        self.model.fit(x,y)

    def predict(self, x):
        '''Predict classes of new data.
        
        Parameters
        ----------
        x : array_like (2d)
            Must have the same number of columns as fitted data.
            Used to predict sky object classes.
        '''
        return self.model.predict(x)

    def predict_proba(self, x):
        '''Predict probabilities for each class.
        
        Parameters
        ----------
        x : array_like (2d)
            Must have the same number of columns as fitted data.
            Used to predict probabilities sky object classes.
        '''
        return self.model.predict_proba(x)
    
    def score(self, x, y):
        '''Return accuracy score.'''
        return self.model.score(x,y)

    def confusion_matrix(self, y_true, y_pred):
        '''Return confusion matrix with predictions.
        
        Parameters
        ----------
        y_true : array_like (1d)
            Array containing the values "GALAXY", "QSO", and/or "STAR".
            True classes of sky objects.

        y : array_like (1d)
            Array containing the values "GALAXY", "QSO", and/or "STAR". 
            Output from predict method: contains class predictons.
        '''
        return confusion_matrix(y_true, y_pred)
