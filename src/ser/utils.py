import os
import sys
import shutil
import logging
import requests


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