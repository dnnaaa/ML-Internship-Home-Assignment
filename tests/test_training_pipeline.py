import pytest
from unittest.mock import MagicMock, patch
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
import pandas as pd

@pytest.fixture
def training_pipeline():
    return TrainingPipeline()

@patch('data_ml_assignment.training.train_pipeline.pd.read_csv')
def test_training_pipeline_init(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        'resume': ['sample resume 1', 'sample resume 2'],
        'label': [1, 2]
    })
    tp = TrainingPipeline()
    assert tp.x_train is not None
    assert tp.x_test is not None
    assert tp.y_train is not None
    assert tp.y_test is not None

@patch('data_ml_assignment.training.train_pipeline.NaiveBayesModel')
def test_train(mock_nb_model, training_pipeline):
    mock_model = MagicMock()
    mock_nb_model.return_value = mock_model
    training_pipeline.train(serialize=False)
    mock_model.fit.assert_called_once()

def test_get_model_performance(training_pipeline):
    training_pipeline.model = MagicMock()
    training_pipeline.model.predict.return_value = [1, 2, 3]
    training_pipeline.y_test = [1, 2, 3]
    accuracy, f1 = training_pipeline.get_model_perfomance()
    assert accuracy == 1.0
    assert f1 == 1.0

@patch('data_ml_assignment.training.train_pipeline.PlotUtils.plot_confusion_matrix')
def test_render_confusion_matrix(mock_plot, training_pipeline):
    training_pipeline.model = MagicMock()
    training_pipeline.model.predict.return_value = [1, 2, 3]
    training_pipeline.y_test = [1, 2, 3]
    training_pipeline.render_confusion_matrix()
    mock_plot.assert_called_once()