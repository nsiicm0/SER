import os

import joblib
import numpy as np
import copy
import io

import dill as pickle # Allows for better serialization
import h5py
import tensorflow as tf
from xgboost import XGBClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
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

    @classmethod
    def load_from_name(cls, save_path: str, name: str, *args, **kwargs):
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
        self.save_path = save_path
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.model_save_path = os.path.join(self.save_path, f'{self.model_name}.pkl')
        _logger.info(f'Instantiated a {self.model_type} model with name {self.model_name}.')
        

    def save_model(self):
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


class SERRandomForestClassifier(RandomForestClassifier, SERBaseModel):
    """
        A simple Random Forest Classifier model.

        Note: We have to specify all variables of the RandomForestClassifier due to a restriction by sklearn. *args and **kwargs doesn't work.
    """
    def __init__(
            self, 
            save_path, 
            name=None, 
            n_estimators=100, *,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.,
            max_features="auto",
            max_leaf_nodes=None,
            min_impurity_decrease=0.,
            min_impurity_split=None,
            bootstrap=True,
            oob_score=False,
            n_jobs=None,
            random_state=None,
            verbose=0,
            warm_start=False,
            class_weight=None,
            ccp_alpha=0.0,
            max_samples=None
        ):
        RandomForestClassifier.__init__(self, 
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples)
        SERBaseModel.__init__(self, save_path, name)


class SERXGBoostClassifier(XGBClassifier, SERBaseModel):
    """
        An XGBoost Classifier model

        Note: We have to specify all variables of the RandomForestClassifier due to a restriction by sklearn. *args and **kwargs doesn't work.
    """
    def __init__(
            self, 
            save_path, 
            name=None, 
            use_label_encoder=True,
            max_depth=None,
            learning_rate=None,
            n_estimators=100,
            verbosity=None,
            objective=None,
            booster=None,
            tree_method=None,
            n_jobs=None,
            gamma=None,
            min_child_weight=None,
            max_delta_step=None,
            subsample=None,
            colsample_bytree=None,
            colsample_bylevel=None,
            colsample_bynode=None,
            reg_alpha=None,
            reg_lambda=None,
            scale_pos_weight=None,
            base_score=None,
            random_state=None,
            missing=np.nan,
            num_parallel_tree=None,
            monotone_constraints=None,
            interaction_constraints=None,
            importance_type="gain",
            gpu_id=None,
            validate_parameters=None
        ):
        XGBClassifier.__init__(
            self, 
            use_label_encoder=use_label_encoder,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            verbosity=verbosity,
            objective=objective,
            booster=booster,
            tree_method=tree_method,
            n_jobs=n_jobs,
            gamma=gamma,
            min_child_weight=min_child_weight,
            max_delta_step=max_delta_step,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            colsample_bylevel=colsample_bylevel,
            colsample_bynode=colsample_bynode,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            scale_pos_weight=scale_pos_weight,
            base_score=base_score,
            random_state=random_state,
            missing=missing,
            num_parallel_tree=num_parallel_tree,
            monotone_constraints=monotone_constraints,
            interaction_constraints=interaction_constraints,
            importance_type=importance_type,
            gpu_id=gpu_id,
            validate_parameters=validate_parameters
        )
        SERBaseModel.__init__(self, save_path, name)


class KerasBaseClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):
    """
    TensorFlow Keras API neural network classifier.

    Workaround the tf.keras.wrappers.scikit_learn.KerasClassifier serialization
    issue using BytesIO and HDF5 in order to enable pickle dumps.

    Adapted from: https://github.com/keras-team/keras/issues/4274#issuecomment-519226139
    Copied from: https://github.com/keras-team/keras/issues/4274#issuecomment-522115115
    """

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


class PickleableKerasClassifier(tf.keras.wrappers.scikit_learn.KerasClassifier):

    def __getstate__(self):
        state = self.__dict__
        model = state['model']
        bio = io.BytesIO()
        with h5py.File(bio) as f:
            model.save(f)
        state['model'] = bio
        return_state = copy.deepcopy(state)
        state['model'] = model
        return return_state

    def __setstate__(self, state):
        with h5py.File(state['model']) as f:
            state['model'] = tf.keras.models.load_model(f)
        self.__dict__ = state


class SERKerasClassifier(SERBaseModel, KerasBaseClassifier):
    """
        A normal Keras Neural Network model
    """
    @classmethod
    def build(cls, n_classes, lr, input_dim):
        clf = Sequential()
        clf.add(Dense(100, activation='relu', input_dim=input_dim))
        clf.add(Dense(n_classes, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        clf.summary(print_fn=_logger.info)
        return clf
    
    def __init__(self, save_path, name=None, *args, **kwargs):
        KerasBaseClassifier.__init__(self, *args, **kwargs)
        SERBaseModel.__init__(self, save_path, name)


class SERKerasDropoutClassifier(SERBaseModel, KerasBaseClassifier):
    """
        A Keras Neural Network model with Dropout
    """
    @classmethod
    def build(cls, n_classes, lr, input_dim, dropout=0.2):
        clf = Sequential()
        clf.add(Dropout(dropout, input_dim=input_dim))
        clf.add(Dense(100, activation='relu'))
        clf.add(Dropout(dropout))
        clf.add(Dense(n_classes, activation='softmax'))
        clf.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr), metrics=['accuracy'])
        clf.summary(print_fn=_logger.info)
        return clf

    def __init__(self, save_path, name=None, *args, **kwargs):
        KerasBaseClassifier.__init__(self, *args, **kwargs)
        SERBaseModel.__init__(self, save_path, name)