import os
import sys
import shutil
import logging
import requests
import numpy as np
import random, string
from typing import Union, Callable


def get_logger(name: str, level: int=logging.INFO):
    """
        Prepares a logger object that can be used for logging information.

        name: str
            The name of the logger.
        level: int
            The logging level for default logging. Defaults to logging.INFO.
    """
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]

    logging.basicConfig(
        level=level, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    logger = logging.getLogger(name)
    return logger


def download_url(url: str, save_path: str, chunk_size=1024) -> str:
    """
        Downloads a file from a given url.

        url: str
            The url to the file. You are responsible that the URL points to a file.
        save_path: str
            The path on the disk where to store the file. If the path is a directory, the filename will be derived from the url.
        chunk_size: int
            The chunksize to be used for downloading. Defaults to 1024.

        Returns: str
            The path to the downloaded file.
    """
    if os.path.isdir(save_path):
        save_path = os.path.join(save_path, os.path.basename(url))
    r = requests.get(url, stream=True)
    if not r.status_code == 200:
        raise IOError(f'Status Code did not match 200. Got: {r.status_code}')
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)
    if not (os.path.exists(save_path) and os.path.isfile(save_path)):
        raise IOError(f'There was an issue writting the file to disk.')
    return save_path


def clean_directory(path: str, exclude: list=['.gitkeep']):
    """
        Cleans a directory.

        path: str
            The path to the folder to be cleaned.
        exclude: list
            The files to exclude from deletion in this folder.
    """
    files = list()
    for f in os.listdir(path):
        if f not in exclude:
            files.append(os.path.join(path, f))
    for f in files:
        try:
            if os.path.isfile(f) or os.path.islink(f):
                os.unlink(f)
            elif os.path.isdir(f):
                shutil.rmtree(f)
        except Exception as e:
            raise IOError(f'Failed to delete {f}. Reason: {e}')


def split_criterion(elements: list, split: Union[Callable, int]=min):
    """
        Helper function return a split value.
        This is used in the dataset preparation.

        elements: list
            The list to do calculations on.
        split: Callable or int
            A callable if we want to calculate on the elements list.
            Else, a static int (will be returned as is).
            Note: if you want to get the average, provide a lambda expression -> lambda x: sum(x)/len(x)
    """
    if callable(split):
        return split(elements)
    elif isinstance(split, int):
        return split
    else:
        raise ValueError(f'Invalid split type provided. Got: {type(split)} - {split}')


def find_peak_amplitude(samples: list):
    """Returns the max amplitude of multiple samples.
    
    samples: list
        A list of pydub.AudioFrame samples.

    Returns a tuple of the max amplitude see more here: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentmax
    """
    max_value = -np.inf
    max_dbfs = -np.inf
    for sample in samples:
        max_value = sample.max if max_value < sample.max else max_value
        max_dbfs = sample.max_dBFS if max_dbfs < sample.max_dBFS else max_dbfs
    return (max_value, max_dbfs)


def one_hot_encode(values: list, unique: list) -> np.ndarray:
    """One-hot encodes values given a unique set of values.

    values: object
        The list of values to one-hot encode
    unique: list
        A set of unique values
    
    """
    if not all(value in unique for value in values):
        raise ValueError(f'Invalide set of values and unique values provided. All values MUST exist in the unique set. ')
    if len(unique) != len(set(unique)):
        raise ValueError(f'The provided unique set of values does not appear to be unique! Got: {unique} -> {set(unique)}')
    if not isinstance(values[0], str):
        raise ValueError(f'Currently only string values supported.')
    # Convert to int_values
    int_values = np.array([list(unique).index(v) for v in values])
    return np.eye(len(unique))[int_values]


def get_random_name(length: int=16) -> str:
    """Generates a random name.
    Used for the model name and the API.
    """
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))
