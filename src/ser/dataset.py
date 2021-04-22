import os
import io
import zipfile
import requests

import pandas as pd

from ser.utils import get_logger, download_url, clean_directory

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


    class _is(object):
        """
            The Condition object is used internally to automate condition checks which guard the functions.
        """
        DOWNLOADED = 'download'
        EXTRACTED = 'extract'
        PREPARED = 'prepare'
    

    DEFAULT_ZIP_NAME = 'download.zip'
    

    def __init__(self, data_path: str, remote_url: str):
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
        self.data = None


    def _check_requirement(self, condition, verbose=True) -> bool:
        """Internal function that checks whether certain conditions are met."""
        _logger.debug(f'Checking condition: {condition}')
        condition_satisfied = False

        if condition == 'download':
            condition_satisfied = os.path.exists(os.path.join(self.pristine_path, Dataset.DEFAULT_ZIP_NAME))
        elif condition == 'extract':
            condition_satisfied = len(list(filter(lambda x: x != '.gitkeep', os.listdir(self.working_path)))) > 0
        elif condition == 'prepare':
            condition_satisfied = isinstance(self.data, pd.DataFrame)

        if condition_satisfied:
            _logger.debug(f'Met conditions for "{condition}"')
            return True
        else:
            if verbose:
                _logger.warning(f'Apparently, the dataset has not yet been {condition}ed. Use .{condition}() to {condition} it first.')
            return False
        

    def download(self, force=False) -> bool:
        if force:
            _logger.info(f'Forcing download. Removes old files from {self.pristine_path}.')
            clean_directory(self.pristine_path)
        # Check if the dataset has been downloaded already. If not, download it.
        if not self._check_requirement(self._is.DOWNLOADED, verbose=False):
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


    def extract(self, force=False) -> bool:
        if force:
            _logger.info(f'Forcing extraction. Removes old files from {self.working_path}.')
            clean_directory(self.working_path)
        # Preliminiary checks
        if not self._check_requirement(self._is.DOWNLOADED): return False
        
        # Check if the dataset has been extracted. If not, do so.
        if not self._check_requirement(self._is.EXTRACTED, verbose=False):
            with zipfile.ZipFile(os.path.join(self.pristine_path, Dataset.DEFAULT_ZIP_NAME), 'r') as zipf:
                zipf.extractall(self.working_path)

        _logger.info('Dataset extracted.')
        return True


    def prepare(self, force=False) -> bool:
        if force:
            _logger.info(f'Forcing preparation. Flushes data variable.')
            self.data = None

        # Preliminary checks
        if not self._check_requirement(self._is.DOWNLOADED): return False
        if not self._check_requirement(self._is.EXTRACTED): return False
        
        # Check if the dataset has been prepared. If not, do so.
        if not self._check_requirement(self._is.PREPARED, verbose=False):
            pass
        
        _logger.info('Dataset prepared.')
        return True


    def get(self) -> pd.DataFrame:
        # Preliminary checks
        if not self._check_requirement(self._is.DOWNLOADED): return None
        if not self._check_requirement(self._is.EXTRACTED): return None
        if not self._check_requirement(self._is.PREPARED): return None
        return self.data


