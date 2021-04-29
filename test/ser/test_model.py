import pytest
import sklearn
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import ser
import os
import shutil
import tempfile

# Common Fixture
@pytest.fixture(scope='module')
def common_path():
    """Source: https://stackoverflow.com/a/62337806"""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    shutil.rmtree(tmpdir)


@pytest.mark.run(order=6)
def test_pipeline_create(common_path):
    pca = PCA(n_components=500)
    clf = RandomForestClassifier(n_estimators=100)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    assert 'name' in pipe.__dict__ and 'save_path' in pipe.__dict__
    assert len(pipe.steps) == 2


@pytest.mark.run(order=7)
def test_pipeline_create_keras(common_path):
    pca = PCA(n_components=2)
    clf = ser.KerasClassifier(build_fn=lambda: ser.KerasClassifier.build_base(n_classes=2, lr=0.001, input_dim=2, dropout=0.2), verbose=0)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    assert 'name' in pipe.__dict__ and 'save_path' in pipe.__dict__
    assert len(pipe.steps) == 2
    assert 'hist' in pipe.named_steps['clf'].__dict__


@pytest.mark.run(order=8)
def test_pipeline_save(common_path):
    pca = PCA(n_components=500)
    clf = RandomForestClassifier(n_estimators=100)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    model_name = pipe.name
    assert not os.path.exists(os.path.join(common_path, f'{model_name}.pkl'))
    pipe.save_model()
    assert os.path.exists(os.path.join(common_path, f'{model_name}.pkl'))


@pytest.mark.run(order=9)
def test_pipeline_save_keras(common_path):
    pca = PCA(n_components=2)
    clf = ser.KerasClassifier(build_fn=lambda: ser.KerasClassifier.build_base(n_classes=2, lr=0.001, input_dim=2, dropout=0.2), verbose=0)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    model_name = pipe.name
    assert not os.path.exists(os.path.join(common_path, f'{model_name}.pkl'))
    pipe.save_model()
    assert os.path.exists(os.path.join(common_path, f'{model_name}.pkl'))


@pytest.mark.run(order=10)
def test_pipeline_restore(common_path):
    pca = PCA(n_components=500)
    clf = RandomForestClassifier(n_estimators=100)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    model_name = pipe.name
    pipe.save_model()
    mdl = ser.Pipeline.load_from_name(save_path=common_path, name=model_name, steps=[])
    assert pipe.name == mdl.name
    assert len(pipe.steps) == len(mdl.steps)


@pytest.mark.run(order=11)
def test_pipeline_restore_keras(common_path):
    pca = PCA(n_components=2)
    clf = ser.KerasClassifier(build_fn=lambda: ser.KerasClassifier.build_base(n_classes=2, lr=0.001, input_dim=2, dropout=0.2), verbose=0)
    pipe = ser.Pipeline(save_path=common_path, steps=[('pca', pca), ('clf', clf)])
    model_name = pipe.name
    pipe.save_model()
    mdl = ser.Pipeline.load_from_name(save_path=common_path, name=model_name, steps=[])
    assert pipe.name == mdl.name
    assert len(pipe.steps) == len(mdl.steps)
    assert isinstance(pipe.named_steps['clf'], ser.KerasClassifier)


