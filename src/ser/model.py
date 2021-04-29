import os

import joblib
import numpy as np
import copy
import io

import dill as pickle # Allows for better serialization
import h5py
import tensorflow as tf
from sklearn.utils.multiclass import unique_labels
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import deserialize, serialize
from tensorflow.python.keras.saving import saving_utils

from ser.utils import get_logger, get_random_name

_logger = get_logger(__name__)


class SERBaseModel(object):
    """The SER Base Model Class.
    
    All Models should inherit from this one (even the ones from sklearn through mixins)
    This way we can enforce a standardized API for saving the models to disk and retrieve them for predictions.

    Idea based on https://scikit-learn.org/stable/developers/develop.html
    """

    @classmethod
    def load_from_name(cls, save_path: str, name: str, *args, **kwargs):
        if issubclass(cls, SERPipeline):
            kwargs['init_pipeline'] = False
        obj = cls(save_path=save_path, name=name, *args, **kwargs)
        obj.restore_model()
        return obj

    def __init__(self, save_path: str, name: str=None):
        """Default constructor
        
        save_path: str
            Folder to store/load model data
        name: str|None
            Name of the model. If you want to reload a model from disk, specify this name.
        """
        self.model_type = self.__class__.__name__
        self.model_name = get_random_name() if name is None else name
        self.name = self.model_name
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_save_path = os.path.join(self.save_path, f'{self.model_name}.pkl')
        _logger.info(f'Instantiated a {self.model_type} model with name {self.model_name}.')
        

    def save_model(self):
        #_logger.info(self.__dict__)
        #_logger.info({k: type(v) for k, v in self.__dict__.items()})
        if os.path.exists(self.model_save_path):
            _logger.warning(f'Found an existing model. Will delete the old one.')
            os.unlink(self.model_save_path)
        pickle.dump(self, open(self.model_save_path, 'wb'))
        _logger.info(f'Saved to {self.model_save_path}')


    def restore_model(self):
        if not os.path.exists(self.model_save_path) and not os.path.isfile(self.model_save_path):
            raise ValueError(f'No model to restore found at {self.model_save_path}')
        restored = pickle.load(open(self.model_save_path, 'rb'))
        # update current object
        self.__dict__.update(restored.__dict__)


class SERPipeline(SKPipeline, SERBaseModel):
    """
        A Pipeline Model

        Note: We have to specify all variables of the Pipeline due to a restriction by sklearn. *args and **kwargs doesn't work.
    """
    def __init__(
            self, 
            save_path: str, 
            steps: list,
            name: str=None, 
            memory: str=None,
            verbose=False,
            init_pipeline=True
        ):
        """Default constructor
        
        save_path: str
            Folder to store/load model data
        steps: list
            The list of pipeline steps - see sklearn.pipeline.Pipeline
        name: str|None
            Name of the model. If you want to reload a model from disk, specify this name.
        memory: str or object with the joblib.Memory interface
            The memory - see sklearn.pipeline.Pipeline
        verbose: bool
            Prints more info while training - see sklearn.pipeline.Pipeline
        init_pipeline: bool
            Calls super to sklearn.pipeline.Pipeline
            Normally set to true if new model get's instantiated, set to False when restored from pkl.
        """
        self.init_pipeline = init_pipeline

        SERBaseModel.__init__(self, save_path=save_path, name=name)
        if self.init_pipeline is None or self.init_pipeline:
            SKPipeline.__init__(self, steps=steps, memory=memory, verbose=verbose)


class KerasBaseClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    Copied from: https://github.com/keras-team/keras/issues/4274#issuecomment-522115115
    """

    def __init__(self, build_fn=None, **sk_params):
        """Default Constructor
            Conditional super function allows for nicer handling within a Pipeline.
        """
        if build_fn is not None:
            super().__init__(build_fn=build_fn, **sk_params)

    def __getstate__(self):
        state = self.__dict__
        if "model" in state:
            model = state["model"]
            model_hdf5_bio = io.BytesIO()
            with h5py.File(model_hdf5_bio, mode="w") as file:
                model.save(file)
            state["model"] = model_hdf5_bio
            state_copy = copy.deepcopy(state)
            state["model"] = model
            return state_copy
        else:
            return state

    def __setstate__(self, state):
        if "model" in state:
            model_hdf5_bio = state["model"]
            with h5py.File(model_hdf5_bio, mode="r") as file:
                state["model"] = tf.keras.models.load_model(file)
        self.__dict__ = state

#
#   Begin Fix to make all Keras models serializable and fix issues with serialization of the TFStack Summary.
#   Copied from: https://github.com/tensorflow/tensorflow/issues/34697#issuecomment-627193883
#   The following functions basically override the tf.keras Model class.
# 
def unpack(model, training_config, weights):
    restored_model = deserialize(model)
    if training_config is not None:
        restored_model.compile(
            **saving_utils.compile_args_from_training_config(
                training_config
            )
        )
    restored_model.set_weights(weights)
    return restored_model


def make_keras_picklable():

    def __reduce__(self):
        model_metadata = saving_utils.model_metadata(self)
        training_config = model_metadata.get("training_config", None)
        model = serialize(self)
        weights = self.get_weights()
        return (unpack, (model, training_config, weights))

    cls = Model
    cls.__reduce__ = __reduce__

#
#   End Fix
#

class SERKerasClassifier(SERBaseModel, KerasBaseClassifier):
    """
        A Keras Neural Network model
    """
    @classmethod
    def build_base(cls, n_classes, lr, input_dim, dropout=0.0, **kwargs):
        clf = Sequential()
        clf.add(Dropout(dropout, input_dim=input_dim))
        clf.add(Dense(100, activation='relu'))
        clf.add(Dropout(dropout))
        clf.add(Dense(n_classes, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        clf.summary(print_fn=_logger.info)
        return clf

    
    def __init__(self, *args, **kwargs):
        make_keras_picklable()
        KerasBaseClassifier.__init__(self, *args, **kwargs)
        self.hist = None

    def fit(self, x, y, **kwargs):
        """
            Overriding .fit() method to retain the history object for later use.
        """
        self.hist = super(KerasBaseClassifier, self).fit(x=x, y=y, **kwargs)
