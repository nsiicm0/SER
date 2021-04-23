import os

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.optimizers import Adam

from ser.utils import get_logger, get_random_name

_logger = get_logger(__name__)


class SERBaseModel(object):
    """The SER Base Model Class.
    
    All Models should inherit from this one (even the ones from sklearn through mixins)

    Idea based on https://scikit-learn.org/stable/developers/develop.html
    """
    def __init__(self, save_path: str, name: str=None):
        """Default constructor
        
        save_path: str
            Folder to store/load model data
        name: str|None
            Name of the model. If you want to reload a model from disk, specify this name.
        """
        self.model_type = self.__class__.__name__
        self.model_name = get_random_name() if name is None else name
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_save_path = os.path.join(self.save_path, f'{self.model_name}.pkl')
        _logger.info(f'Instantiated a {self.model_type} model with name {self.model_name}.')
        

    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()


    def save_model(self):
        if os.path.exists(self.model_save_path):
            _logger.warning(f'Found an existing model. Will delete the old one.')
            os.unlink(self.model_save_path)
        joblib.dump(self, self.model_save_path)


    def restore_model(self):
        if not os.path.exists(self.model_save_path) and not os.path.isfile(self.model_save_path):
            raise ValueError(f'No model to restore found at {self.model_save_path}')
        restored = joblib.load(self.model_save_path)
        # update current object
        self.__dict__.update(restored.__dict__)


class SERRandomForestClassifier(RandomForestClassifier, SERBaseModel):
    """
        A simple Random Forest Classifier model.
    """
    pass


class SERXGBoostClassifier(XGBClassifier, SERBaseModel):
    """
        An XGBoost Classifier model
    """
    pass


class SERKerasClassifier(KerasClassifier, SERBaseModel):
    """
        A Keras Neural Network model
    """
    @classmethod
    def build(cls, n_classes, lr, input_dim):
        clf = Sequential()
        clf.add(Dense(1024, activation='relu', input_dim=input_dim))
        clf.add(Dense(256, activation='relu'))
        clf.add(Dense(100, activation='relu'))
        clf.add(Dense(n_classes, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        return clf
