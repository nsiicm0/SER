import os
import base64
from sklearn.model_selection import train_test_split
import dill as pickle # Allows for better serialization

from ser.exception import TrainerException
from ser.utils import get_logger, one_hot_encode
from ser import Sample, Dataset, MFCCFeatureExtractor, SpeakerAndGenderAndTextTypeFeatureExtractor, KerasClassifier, KerasDropoutClassifier

_logger = get_logger(__name__)


class Trainer(object):
    """
        A Trainer object that handles the whole training lifecycle.
        See Predictor for prediction cycle.

        config: dict
            A config dict containing the dataset and model configuration. Should be of the following form:
            {
                'data_path': <path to data folder>,             # mandatory, this is where the data will be saved into
                'remote_url': <path to the remote file>,        # mandatory, this is where the data zip resides on the remote host
                'random_state': <int>                           # optional, a random state used for seeding RNGs                         
                'dataset_force': True|False,                    # optional, would force all function calls in dataset
                'dataset_load_split_at': 'min|max|avg|<int>',   # optional, see ser.Dataset.load() for more information
                'dataset_load_normalize': True|False,           # optional, see ser.Dataset.load() for more information
                'dataset_feature_extract_methods': {            # optional, see ser.Dataset.feature_extract() for more information
                    'METHOD_NAME': <CONFIG>,                    # METHOD_NAME accepts a valid feature extractor, currently supported: SpeakerAndGenderAndTextType|MFCC
                    ...                                         # CONFIG is the config dict as found in ser.Dataset.feature_extract()
                },
                'model_type': <MODEL NAME>,                     # mandatory, currently supported: KerasClassifier|KerasDropoutClassifier
                'model_save_path': <path to model folder>       # mandatory, this is where the model will be saved into (also will hold other training related data)
                'model_config': <MODEL TRAINING CONFIG>         # optional, elsewise default values will be used.
                                                                # Default config: {
                                                                #   'KerasClassifier': {'lr': 0.001, 'epochs': 20, 'batch_size': 32},
                                                                #   'KerasDropoutClassifier': {'lr': 0.001, 'epochs': 100, 'batch_size': 32, 'dropout': 0.2},
                                                                # }
            }
    """

    def __init__(self, config: dict):
        self.config = config
        self.dataset = None
        if 'dataset_force' in self.config:
            self.force = self.config['dataset_force']
        else:
            self.force = False
        self.model = None
        self.model_default_config = {
            'KerasClassifier': {'lr': 0.001, 'epochs': 20, 'batch_size': 32},
            'KerasDropoutClassifier': {'lr': 0.001, 'epochs': 100, 'batch_size': 32, 'dropout': 0.2},
        }
        if self.config['model_config'] is None:
            self.model_config = self.model_default_config[self.config['model_type']]
        else:
            if 'lr' not in self.config['model_config'] or 'epochs' not in self.config['model_config'] or 'batch_size' not in self.config['model_config']:
                raise TrainerException(f'Invalid model config provided. Make sure to include "lr", "epochs", and "batch_size". Got: {self.config["model_config"]}')
            self.model_config = self.config['model_config']
        self.model_id = None
        self.X, self.y, self.N, self.X_train, self.X_test, self.y_train, self.y_test, self.N_train, self.N_test, self.y_test_onehot = None, None, None, None, None, None, None, None, None, None
        self.hist = None
    

    def prepare_dataset(self) -> None:
        """
            Loads and prepares the dataset routine.
        """
        if 'data_path' not in self.config: raise TrainerException('No "data_path" provided in config.')
        if 'remote_url' not in self.config: raise TrainerException('No "remote_url" provided in config.')

        _logger.info('Preparing dataset object')
        self.dataset = Dataset(data_path=self.config['data_path'], remote_url=self.config['remote_url'])

        _logger.info('Downloading data')
        self.dataset.download(force=self.force)

        _logger.info('Extracting the data')
        self.dataset.extract(force=self.force)

        _logger.info('Parsing the samples in the dataset')
        self.dataset.prepare(force=self.force)

        _logger.info('Loading the data into memory')
        _load_args = dict({'force': self.force})
        if 'dataset_load_split_at' in self.config and self.config['dataset_load_split_at'] is not None:
            if self.config['dataset_load_split_at'] == 'min':
                _load_args['split_at'] = min
            elif self.config['dataset_load_split_at'] == 'max':
                _load_args['split_at'] = max
            elif self.config['dataset_load_split_at'] == 'avg':
                _load_args['split_at'] = lambda x: sum(x)/len(x)
            elif self.config['dataset_load_split_at'].isdigit():
                _load_args['split_at'] = int(self.config['dataset_load_split_at'])
            else:
                raise TrainerException('Invalid option provided for "dataset_load_split_at"')
        if 'dataset_load_normalize' in self.config and self.config['dataset_load_normalize'] is not None:
            _load_args['normalize'] = self.config['dataset_load_normalize']       
        self.dataset.load(**_load_args)
        
        _logger.info('Feature extracting.')
        _fe_args = dict({'force': self.force})
        if 'dataset_feature_extract_methods' in self.config and self.config['dataset_feature_extract_methods'] is not None:
            _methods = dict({})
            for requested_method, config in dict(self.config['dataset_feature_extract_methods']).items():
                if requested_method == 'SpeakerAndGenderAndTextType':
                    _methods[requested_method] = (SpeakerAndGenderAndTextTypeFeatureExtractor, config)
                elif requested_method == 'MFCC':
                    _methods[requested_method] = (MFCCFeatureExtractor, config)
                else:
                    raise TrainerException(f'Encountered invalid feature extractor request. Got: {requested_method} - {config}')
            _fe_args['extraction_methods'] = _methods
        self.dataset.feature_extract(**_fe_args)

        self.X, self.y, self.N = self.dataset.get_features(return_names=True)
        _logger.info('Done with dataset loading.')


    def prepare_model(self) -> None:
        """
            Sets the model up for training.
        """
        if not os.path.exists(self.config['model_save_path']):
            _logger.info(f'{self.config["model_save_path"]} does not yet exist.')
            os.makedirs(self.config['model_save_path'])
        if self.config['model_type'] == 'KerasClassifier':
            self.model = KerasClassifier(
                save_path=self.config['model_save_path'], 
                build_fn=lambda: KerasClassifier.build(len(Sample.EMOTIONS), self.model_config['lr'], self.X.shape[1]), 
                epochs=self.model_config['epochs'], 
                batch_size=self.model_config['batch_size'], 
                verbose=0
            )
        elif self.config['model_type'] == 'KerasDropoutClassifier':
            self.model = KerasDropoutClassifier(
                save_path=self.config['model_save_path'], 
                build_fn=lambda: KerasDropoutClassifier.build(len(Sample.EMOTIONS), self.model_config['lr'], self.X.shape[1], self.model_config['dropout']), 
                epochs=self.model_config['epochs'], 
                batch_size=self.model_config['batch_size'], 
                verbose=0
            )
        else:
            raise TrainerException(f'An invalid model type was provided. Got: {self.config["model_type"]}')
        self.model_id = self.model.model_name
        _logger.info(f'Prepared model with ID {self.model_id}')


    def train(self) -> None:
        """
            Trains the model.
        """
        _logger.info('Starting training.')
        self.X_train, self.X_test, self.y_train, self.y_test, self.N_train, self.N_test = train_test_split(self.X, self.y, self.N, test_size=0.1, random_state=self.config['random_state'])
        self.y_test_onehot = one_hot_encode(self.y_test, Sample.EMOTIONS.keys())
        _logger.info(f'Data shape:\n\tX_train: {self.X_train.shape}\n\ty_train: {self.y_train.shape}\n\tX_test: {self.X_test.shape}\n\ty_test: {self.y_test.shape}')
        self.hist = self.model.fit(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test_onehot))
        _logger.info('Done with model training.')
    

    def save(self) -> None:
        """
            Save model and trainer config.
        """
        _logger.info('Saving model')
        # saving the model
        self.model.save_model()
        # saving the trainer state
        _logger.info('Saving trainer state')
        trainer_config_path = os.path.join(self.config['model_save_path'], f'{self.model_id}_config.pkl')
        if os.path.exists(trainer_config_path):
            _logger.warning(f'Found an existing config. Will delete the old one.')
            os.unlink(trainer_config_path)
        obj = dict({
            'config': dict(self.config),
            'id': self.model_id,
            'model_config': self.model_config,
            'model_type': self.config['model_type'],
            'classes': [f'{clazz} - {Sample.EMOTIONS[clazz]}' for clazz in self.model.classes_.tolist()],
            'X_test': self.X_test.tolist(),
            'y_test': self.y_test.tolist(),
            'y_test_onehot': self.y_test_onehot.tolist(),
            'sample_test': self.N_test.tolist()
        })
        pickle.dump(obj, open(trainer_config_path, 'wb'))
        _logger.info('Done saving!')
