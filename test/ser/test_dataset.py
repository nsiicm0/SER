import os
import shutil
import tempfile
import pathlib
import pytest
import numpy as np
import ser
import copy
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


# Common Fixture
@pytest.fixture(scope='module')
def common_path():
    """Source: https://stackoverflow.com/a/62337806"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture(scope='module')
def common_dataset(common_path):
    dataset = ser.Dataset(data_path=common_path, remote_url='http://emodb.bilderbar.info/download/download.zip')
    yield dataset
    del dataset

# test_download
def test_download(common_dataset):
    pristine_path = common_dataset.pristine_path
    # check for empty directory
    assert len(list(filter(lambda x: x not in ['.gitkeep'], os.listdir(pristine_path)))) == 0
    common_dataset.download()
    # make sure the file has been downloaded
    assert 'download.zip' in os.listdir(pristine_path)
    mod_time = os.path.getmtime(os.path.join(pristine_path, 'download.zip'))
    common_dataset.download()
    # make sure the file timestamp is the same
    mod_time2 = os.path.getmtime(os.path.join(pristine_path, 'download.zip'))
    assert mod_time == mod_time2
    common_dataset.download(force=True)
    mod_time3 = os.path.getmtime(os.path.join(pristine_path, 'download.zip'))
    assert mod_time != mod_time3


# test_extract
def test_extract(common_dataset):
    pristine_path = common_dataset.pristine_path
    working_path = common_dataset.working_path
    # check for empty directory
    assert len(list(filter(lambda x: x not in ['.gitkeep'], os.listdir(working_path)))) == 0
    common_dataset.extract()
    # make sure the file has been extracted
    assert len(list(filter(lambda x: x not in ['.gitkeep'], os.listdir(working_path)))) > 0
    # make sure the files don't get reextracted twice
    mod_time = os.path.getmtime(os.path.join(working_path, 'erklaerung.txt'))
    common_dataset.extract()
    mod_time2 = os.path.getmtime(os.path.join(working_path, 'erklaerung.txt'))
    assert mod_time == mod_time2
    # test force function
    common_dataset.extract(force=True)
    mod_time3 = os.path.getmtime(os.path.join(working_path, 'erklaerung.txt'))
    assert mod_time != mod_time3
    

# test_prepare
def test_prepare(common_dataset):
    # check if is not prepared
    assert common_dataset._Dataset__data is None
    # check is prepared
    common_dataset.prepare()
    assert common_dataset._Dataset__data is not None
    # check if is sample
    assert isinstance(common_dataset._Dataset__data[0], ser.Sample)


# test_load
def test_load(common_dataset):
    # check if is not loaded
    assert common_dataset._Dataset__loaded_data is None
    # check is prepared
    common_dataset.load()
    assert common_dataset._Dataset__loaded_data is not None
    # check if is sample
    assert isinstance(common_dataset._Dataset__loaded_data[0], ser.Sample)


# test_feature_extract
def test_feature_extract(common_dataset):
    # check if is not feature extracted
    assert common_dataset.N is None and common_dataset.X is None and common_dataset.y is None
    # check is prepared
    common_dataset.feature_extract()
    assert common_dataset.N is not None and common_dataset.X is not None and common_dataset.y is not None
    # check if is ndarray
    assert isinstance(common_dataset.N, np.ndarray) and isinstance(common_dataset.X, np.ndarray) and isinstance(common_dataset.y, np.ndarray)

