import os
import zipfile
import requests
from typing import Tuple, Union, Callable

import numpy as np
import pandas as pd

import ser
from ser.utils import get_logger, download_url, clean_directory, split_criterion, find_peak_amplitude

_logger = get_logger(__name__)


class Dataset(object):
    """
        Dataset class

        This wrapper class is designed to handle all tasks related to the dataset.
        
        data_path: str
            The base path to the dataset. Usually, this should be './data' from the project root.
        remote_url: str
            The URL for the zip file to download.
    """


    class __is(object):
        """
            The Condition object is used internally to automate condition checks which guard the functions.
        """
        DOWNLOADED = 'download'
        EXTRACTED = 'extract'
        PREPARED = 'prepare'
        LOADED = 'load'
        FEATURE_EXTRACTED = 'feature extract'
    

    DEFAULT_ZIP_NAME = 'download.zip'
    

    def __init__(
            self, 
            data_path: str, 
            remote_url: str
        ):
        """Constructor
        data_path: str
            Path to the data folder. Should default normally to ./data in the project.

        remoute_url: str
            The URL to download the ZIP file from. Must point directly to the ZIP file.
        """
        _logger.info('Creating Dataset Wrapper object.')
        # Check and create the base path if it does not exist
        self.base_path = data_path
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path)
        _logger.info(f'> Base Path at "{self.base_path}"')
        
        # Check and create the pristine path if it does not exist
        self.pristine_path = os.path.join(self.base_path, 'pristine')
        if not os.path.exists(self.pristine_path):
            os.makedirs(self.pristine_path)
        _logger.info(f'> Pristine Path at "{self.pristine_path}"')

        # Check and create the working path if it does not exist
        self.working_path = os.path.join(self.base_path, 'working')
        if not os.path.exists(self.working_path):
            os.makedirs(self.working_path)      
        _logger.info(f'> Working Path at "{self.working_path}"')  

        # Remote URL
        self.remote_url = remote_url
        _logger.info(f'Make sure that the {self.remote_url} points to a ZIP file.')

        # The dataset
        self.__data = None
        self.__loaded_data = None
        self.X = None
        self.y = None
        self.N = None

        self.normalization_max_value = 0

    def __check_requirements(
            self, 
            conditions: list
        ) -> bool:
        """Internal function that checks whether multiple conditions are met."""
        for condition in conditions:
            successful = self.__check_requirement(condition)
            if not successful: return False
        return True

    def __check_requirement(
            self, 
            condition: str, 
            verbose: bool=True
        ) -> bool:
        """Internal function that checks whether a certain condition is met."""
        _logger.debug(f'Checking condition: {condition}')
        condition_satisfied = False

        if condition == 'download':
            condition_satisfied = os.path.exists(os.path.join(self.pristine_path, Dataset.DEFAULT_ZIP_NAME))
        elif condition == 'extract':
            condition_satisfied = len(list(filter(lambda x: x != '.gitkeep', os.listdir(self.working_path)))) > 0
        elif condition == 'prepare':
            condition_satisfied = isinstance(self.__data, list)
        elif condition == 'load':
            condition_satisfied = isinstance(self.__loaded_data, list)
        elif condition == 'feature extract':
            condition_satisfied = isinstance(self.X, np.ndarray) and isinstance(self.y, np.ndarray)
        else:
            raise ValueError(f'Invalid condition was provided. Got: {condition}')

        if condition_satisfied:
            _logger.debug(f'Met conditions for "{condition}"')
            return True
        else:
            if verbose:
                _logger.warning(f'Apparently, the dataset has not yet been {condition}ed. Use .{condition.replace(" ", "_")}() to {condition} it first.')
            return False
        

    def clean(self) -> None:
        """Cleans the data directory."""
        _logger.info(f'Cleaning the pristine directory. {self.pristine_path}')
        clean_directory(self.pristine_path)
        _logger.info(f'Cleaning the working directory. {self.working_path}')
        clean_directory(self.working_path)


    def download(
            self, 
            force: bool=False,
            **kwargs
        ) -> bool:
        """Download function to download the data specified at the remote_url.

        force: bool
            Forces redownload of the data. Function would not redownload if the data already exists.
        """
        if force:
            _logger.info(f'Forcing download. Removes old files from {self.pristine_path}.')
            clean_directory(self.pristine_path)
        # Check if the dataset has been downloaded already. If not, download it.
        if not self.__check_requirement(self.__is.DOWNLOADED, verbose=False):
            # Download
            try:
                downloaded_file = download_url(self.remote_url, os.path.join(self.pristine_path, Dataset.DEFAULT_ZIP_NAME))
            except IOError as e:
                _logger.exception('An exception occured during downloading of the file')
            except:
                _logger.exception('An unhandled excpetion occured.')
            else:
                _logger.info(f'Successfully downloaded file to {downloaded_file}')

        _logger.info('Dataset downloaded.')
        return True


    def extract(
            self, 
            force: bool=False,
            **kwargs
        ) -> bool:
        """Extract function to extract the ZIP file into the working folder.

        force: bool
            Forces reextraction of the data. Function would not reextract if the data has already been extracted.
        """
        if force:
            _logger.info(f'Forcing extraction. Removes old files from {self.working_path}.')
            clean_directory(self.working_path)
        # Preliminiary checks
        if not self.__check_requirements([self.__is.DOWNLOADED]): return False
        
        # Check if the dataset has been extracted. If not, do so.
        if not self.__check_requirement(self.__is.EXTRACTED, verbose=False):
            with zipfile.ZipFile(os.path.join(self.pristine_path, Dataset.DEFAULT_ZIP_NAME), 'r') as zipf:
                zipf.extractall(self.working_path)

        _logger.info('Dataset extracted.')
        return True


    def prepare(
            self, 
            force: bool=False,
            **kwargs
        ) -> list:
        """Prepare function to wrap all files in a Sample object.

        force: bool
            Forces preparation of the data. Function would not reprepare if the data has already been prepared.
        """
        if force:
            _logger.info(f'Forcing preparation. Flushes data variable.')
            self.__data = None

        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED]): return False
        
        # Check if the dataset has been prepared. If not, do so.
        if not self.__check_requirement(self.__is.PREPARED, verbose=False):
            files = list(map(lambda item: os.path.join(self.working_path, 'wav', item), os.listdir(os.path.join(self.working_path, 'wav'))))
            self.__data = list(ser.Sample.from_list(files))
        
        _logger.info('Dataset prepared.')
        return True

    
    def load(
            self, 
            split_at: Union[Callable, int, None]=min, 
            normalize: bool=True, 
            force: bool=False,
            **kwargs
        ) -> bool:
        """Load function to load the wav files to the RAM.
        The function also provides options to split the files into equally sized chunks (incl. padding) and normalization of the audio.

        split_at: Callable|int|None
            If a function is provided such as "min", "max" or a lambda expression, it will be evaluated to define a chunk length in the samples.
            If an int is provided, that value will be used.
            If None is provided, nothing will be split or padded.
        
        normalize: bool
            Normalizes the audio. The max value will be stored in self.normalization_max_value of the dataset.

        force: bool
            Forces reloading of the data. Function would not reload if the data has already been loaded.
        """
        if force:
            _logger.info(f'Forcing loading. Flushes loaded data variable.')
            self.__loaded_data = None
        
        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED, self.__is.PREPARED]): return False

        # Check if the dataset has been loaded. If not, do so.
        if not self.__check_requirement(self.__is.LOADED, verbose=False):
            # First load all the data
            data = list([sample.load_from_disk() for sample in self.__data])
            if split_at is not None:
                # Then we make sure all data is of the same length as models usually don't like varying input length.
                ## calculate split length
                split_length = split_criterion([sample.get('length') for sample in data], split_at)
                ## apply padding
                self.__loaded_data = list()
                for sample in data:
                    self.__loaded_data.extend(sample.split_and_pad(split_length))
            else:
                self.__loaded_data = data
            if normalize:
                _logger.info('Normalizing requested.')
                ## get max amplitude
                self.normalization_max_value, _ = find_peak_amplitude([sample.get('object') for sample in self.__loaded_data])
                ## normalize
                for sample in self.__loaded_data:
                    sample.normalize(self.normalization_max_value)
        
        _logger.info('Dataset loaded.')
        return True


    def get_normalization_information(self) -> int:
        """Returns the normalization value used to normalize a sample."""
        return self.normalization_max_value

    
    def feature_extract(
            self, 
            extraction_methods: dict={
                'SpeakerAndGenderAndTextType': (ser.SpeakerAndGenderAndTextTypeFeatureExtractor, {'speaker': True, 'gender': True, 'text': True}), 
                'MFCC': (ser.MFCCFeatureExtractor, {'num_cepstral': 13})}, 
            force: bool=False,
            **kwargs
        ) -> bool:
        """Feature Extraction function to extract features from the audio.

        extraction_methods: dict
            This is a dict of the following format (note: the name is only used for printing the progress):
            {
                NAME1: (<Feature Extraction Class that implements ser.FeatureExtractor>, <Config for Feature Extractor - see Feature Extractor Help for more information.>),
                ...
            }
            The function will iterate over all provided extractors (order is provided by dict data structure).
            
        force: bool
            Forces feature reextraction of the data. Function would not feature reextract if the data has already been feature extracted.
        """
        if force:
            _logger.info(f'Forcing feature extraction. Flushes X, y variable.')
            self.X, self.y, self.N = None, None, None
        
        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED, self.__is.PREPARED, self.__is.LOADED]): return False

        # Check if the features have been extracted. If not, do so.
        if not self.__check_requirement(self.__is.FEATURE_EXTRACTED, verbose=False):
            features = list()
            targets = list()
            for name, (clazz, kwargs) in extraction_methods.items():
                _logger.info(f'Now extracting features using {name} Feature Extractor')
                _X, _y = clazz.extract(self.__loaded_data, **kwargs)
                features.append(_X)
                targets.append(_y)
            # check whether we got the same amount of features in all extractions
            if len(set([len(f) for f in features])) > 1: raise ValueError('There was an issue with the feature extraction. Not all feature extractors return the same amount samples.')
            # combine the features for each sample, so we get one np.array for each sample with all different features
            combined_features = list()
            for _features in zip(*features):
                combined_features.append(np.concatenate(_features))
            self.N = np.array([sample.get('name') for sample in self.__loaded_data])
            self.X = np.array(combined_features)
            self.y = np.array(targets)[:1][0] # since we already made sure that extractions provided the same amount of samples, we just grab the first target set
            
        _logger.info('Feature extracted.')
        return True


    def get_features(self, return_names: bool=False) -> Tuple[tuple, tuple]:
        """Returns the Tuple of X and y values.
        
        return_names: bool
            Returns the names of the samples too.
        """
        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED, self.__is.PREPARED, self.__is.LOADED, self.__is.FEATURE_EXTRACTED]): return False

        if return_names:
            return self.X, self.y, self.N
        else:
            return self.X, self.y

    def get_raw_data(self) -> list:
        """Returns the raw data after it has been prepared."""
        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED, self.__is.PREPARED]): return False

        return self.__data

    def get_loaded_data(self) -> list:
        """Returns the data after it has been loaded from disk."""
        # Preliminary checks
        if not self.__check_requirements([self.__is.DOWNLOADED, self.__is.EXTRACTED, self.__is.PREPARED, self.__is.LOADED]): return False

        return self.__loaded_data

