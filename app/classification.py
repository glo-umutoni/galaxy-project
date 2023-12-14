'''Module used to classify data from Sloan Digital Sky Survey as Star, Galaxy, or QSO.

This module contains a classifier following the SKlearn API.  This class object can 
be fit with a single preprocessed dataframe containing sky object information where
each row corresponds to one object, and a list containing their corresponding
classes ('STAR', 'GALAXY', or 'QSO').  The class of new data can be predicted from
this as long as the new data follows the same format as the fitted data.  
The quality of predictions can be assessed using the accuracy score returned by the
`score` method or by a confusion matrix generated from the `confusion_matrix` method.

Possible input data could include aligned wavelengths, metadata, or a concatenation 
of the two. 
'''

from astroquery.sdss import SDSS
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

class Classifier:
    '''Classification estimator adapted from sklearn models.
    
    Supported classifiers include K-Neighbors Classification,
    Logistic Regression, and Random forest classificaton.

    Attribute
    ----------
    MODELS: dict
        contains the models the user can use for classification.

    Example
    ----------
    Possible usage of classifier
        >>> Classifier(model_name='LogisticRegression')
        >>> query="SELECT TOP 20 SpecObjID, ra, dec, z, run2d, class FROM SpecObj"
        >>> data = Data()
        >>> data.extract_from_query(query)
        >>> # align spectra 
        >>> object_ids = list(data.data['SpecObjID'])
        >>> _, aligned_spectra = WavelengthAlignment.align(object_ids, min_val=1, max_val=3, num_points=10)
        >>> # preprocess data
        >>> std_spectra = Preprocessing.normalize(data=pd.DataFrame(aligned_spectra))
        >>> # drop class and ID
        >>> metadata = data.data.drop(columns = ['class', 'SpecObjID'])
        >>> std_metadata = Preprocessing.normalize(data=metadata)
        >>> # combine metadata and spectra
        >>> X = pd.concat([std_metadata, std_spectra],axis=1).to_numpy()
        >>> y = data.data['class'].apply(lambda x : str(x))
        >>> # perform object prediction
        >>> classifier = Classifier('LogisticRegression')
        >>> classifier.fit(X,y)
        >>> y_pred = classifier.predict(X)
    '''

    MODELS = {
            "KNeighborsClassifier":KNeighborsClassifier,
            "LogisticRegression":LogisticRegression,
            "RandomForestClassifier":RandomForestClassifier
        }

    def __init__(self, model_name, **kwargs):
        '''Initialize estimator object from sklearn. Specify classification algorithm.

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
        
        if model_name not in {key for key in Classifier.MODELS}:
            raise ValueError("Must be one of 'KNeighborsClassifier','LogisticRegression','RandomForestClassifier'")
        
        self.model = Classifier.MODELS[model_name](**kwargs)

    def fit(self, x, y):
        '''Fit classifier.
        
        Parameters
        ----------
        x : array_like (2d)
            Spectral and/or metadata information for sky objects. 
            Should already be preprocessed.  Each row corresponds to one sky object.

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

        Returns 
        ----------
        y_pred : array_like (1d)
            Array containing the values "GALAXY", "QSO", and/or "STAR", 
            the predicted classes for the sky objects according to 
            the fitted classifier.
        '''
        return self.model.predict(x)

    def predict_proba(self, x):
        '''Predict probabilities for each class.
        
        Parameters
        ----------
        x : array_like (2d)
            Must have the same number of columns as fitted data.
            Used to predict probabilities sky object classes.

        Returns 
        ----------
        y_pred_proba : array_like (2d)
            Array of shape (x.shape[0], 3) containing the 
            predicted probabilities of the sky object belonging
            to each class, as assigned by the classifier. 
        '''
        return self.model.predict_proba(x)
    
    def score(self, x, y):
        '''Return accuracy score.
        
        Parameters
        ----------
        x : array_like (2d)
            Must have the same number of columns as fitted data.
        
        y : array_like (1d)
            Array containing the values "GALAXY", "QSO", and/or "STAR", 
            the true classes for the sky objects in x.  

        Returns 
        ----------
        accuracy : float
            Classification accuracy of fitted classifier predicting
            the classes of the input x data. Value between 0 and 1. 
        '''
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

        Returns 
        ----------
        cm : array_like (2d)
            Confusion matrix of shape (n_classes, n_classes). 
        '''
        return confusion_matrix(y_true, y_pred)
    
    @staticmethod
    def data_for_classifier(data, merge_data = None):
        '''Combine metadata and spectral data.

        Parameters
        ----------
        data : pd.DataFrame
            Pandas dataframe containing features and 'class' column
            for subsequent classification.  This could be the data contained in the
            Data.data attribute (ensuring that it is subsetted according
            to relevant features for classification).  Each row corresponds
            to one sky object. 

        merge_data : array_like (2d)
            This could be spectra output from wavelength_alignment module.  
            Each row corresponds to spectra from one sky object.  
            Although this is intended for spectra, this 
            in reality could take any data that could be used as additional features.

        Returns 
        ----------
        X : pd.DataFrame (2d)
            Dataframe that has merged 'data' and 'spectra' along axis 1 
            (spectra appended as columns) 

        y : array-like (1d)
            Contains response variable information. 

        Raises
        ----------
        ValueError
            Raised if data is not a pandas DataFrame
            Raised if data is empty
            Raised if 'class' column is not found in data
        '''

        if not isinstance(data, pd.DataFrame): 
            raise ValueError("data is not a pd.DataFrame")
        if data.empty: 
            raise ValueError("data is empty, make sure you extract data first.")
        if not 'class' in data.columns:
            raise ValueError("class column is not in data. Check your query")

        # define y
        y = data['class'].apply(lambda x : str(x.decode()))
        # drop class from features
        data = data.drop(columns = ['class'])

        if merge_data:
            # ensure merge_data is a dataframe
            merge_data = pd.DataFrame(merge_data)
            # check that column names are strings
            if not all(isinstance(col, str) for col in merge_data.columns):
                merge_data.columns = [str(col) for col in merge_data.columns]

            return pd.concat([data, merge_data],axis=1), y
        
        else:
            return data, y