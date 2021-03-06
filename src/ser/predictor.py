import os
import dill as pickle

import numpy as np

from ser.exception import PredictorException
from ser.utils import get_logger
from ser import Pipeline

_logger = get_logger(__name__)


class Predictor(object):
    """
        A Predictor object that takes over the predicting lifecycle after training.
        See Trainer for training cycle.

        model_save_path: <path to model folder>
            The path holds the pickles of the model and the training config
        model_id: str
            The ID of the train cycle.
        sample_name: str
            The name of the sample to predict. Must be part of the validation set.
    """

    def __init__(self, model_save_path: str, model_id: str, sample_name: str):
        self.model_save_path = model_save_path
        self.model_id = model_id
        self.sample_name = sample_name
        self.training_conf = None
        self.model = None


    def restore(self) -> None:
        """
            Restores a previous model for predicting.
        """
        _logger.info(f'Attempting to restore model with id {self.model_id}.')
        config_path = os.path.join(self.model_save_path, f'{self.model_id}_config.pkl')
        model_path = os.path.join(self.model_save_path, f'{self.model_id}.pkl')
        if not os.path.exists(config_path): raise PredictorException(f'There is no training config for the model ID: {self.model_id}')
        if not os.path.exists(model_path): raise PredictorException(f'There is no model for the model ID: {self.model_id}')
        # load the config
        _logger.info('Loading training config')
        self.training_conf = pickle.load(open(config_path, 'rb'))
        # load the model
        _logger.info('Loading model')
        self.model = Pipeline.load_from_name(save_path=self.model_save_path, name=self.model_id, steps=[])
        _logger.info('Restore done.')


    def predict(self) -> dict:
        """
            Function for predicting a sample.
        """
        _logger.info(f'Predicting sample: {self.sample_name}')
        # Looking up index
        try:
            idx = list()
            for name in self.sample_name:
                idx.append(self.training_conf['sample_test'].index(name))
        except ValueError:
            raise PredictorException(f'Provided sample was not part of the validation set. Got: {self.sample_name}')
        # Get features and target
        X = np.array(self.training_conf['X_test'])[idx]
        y = np.array(self.training_conf['y_test'])[idx]
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)

        response = dict({
            'sample_name': self.sample_name,
            'ground_truth': y.tolist(),
            'prediction': y_pred.tolist(),
            'probabilities': [{k: v for k, v in zip(self.training_conf['classes'], proba)} for proba in y_pred_proba.tolist()]
        })
        return response
    