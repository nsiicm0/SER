import json
import pytest
from api import create_app
import os
import shutil
import tempfile

@pytest.fixture(scope="module")
def app():
    app = create_app()
    return app

@pytest.fixture(scope="module")
def common_path():
    """Source: https://stackoverflow.com/a/62337806"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)

@pytest.fixture(scope="module")
def data_path(common_path):
    return common_path

@pytest.fixture(scope="module")
def model_path(common_path):
    return common_path


@pytest.mark.run(order=12)
def test_main(client):
    res = client.get('/')
    assert res.status_code == 200
    assert 'Status' in dict(res.json)
    assert 'Available Endpoints' in dict(res.json)

@pytest.mark.run(order=13)
def test_train_predict(client, data_path, model_path):
    payload = {
        "data_path": data_path,
        "dataset_feature_extract_methods": "{\"MFCC\": {\"num_cepstral\": 13}}",
        "dataset_force": False,
        "model_save_path": model_path,
        "model_type": "KerasClassifier",
        "remote_url": "http://emodb.bilderbar.info/download/download.zip"
    }
    res = client.post('/train', headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
    assert res.status_code == 200
    assert 'Model ID' in dict(res.json)
    assert 'Test Samples' in dict(res.json)
    payload = {
        "model_id": dict(res.json)['Model ID'],
        "model_save_path": model_path,
        "sample_name": dict(res.json)['Test Samples'][0]
    }
    res = client.post('/predict', headers={'Content-Type': 'application/json'}, data=json.dumps(payload))
    assert res.status_code == 200
    assert 'sample_name' in dict(res.json)
    assert 'ground_truth' in dict(res.json)
    assert 'prediction' in dict(res.json)
    assert 'probabilities' in dict(res.json)
