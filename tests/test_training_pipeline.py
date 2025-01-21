import pytest
from unittest.mock import MagicMock, patch
from data_ml_assignment.training.train_pipeline import TrainingPipeline
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

@patch('data_ml_assignment.training.train_pipeline.GridSearchCV')
@patch('data_ml_assignment.training.train_pipeline.LogisticRegression')
def test_train(mock_lr_model, mock_grid_search, training_pipeline):
    mock_model = MagicMock()
    mock_lr_model.return_value = mock_model

    mock_grid_search_instance = MagicMock()
    mock_grid_search.return_value = mock_grid_search_instance

    mock_grid_search_instance.best_estimator_ = mock_model

    training_pipeline.train(serialize=False)

    mock_grid_search_instance.fit.assert_called_once()

    assert training_pipeline.model == mock_model

def test_get_model_performance(training_pipeline):
    training_pipeline.vectorizer = MagicMock()
    training_pipeline.vectorizer.transform.return_value = [[1, 2, 3], [4, 5, 6]]  # Mock transformed data
    training_pipeline.model = MagicMock()
    training_pipeline.model.predict.return_value = [1, 2]
    training_pipeline.y_test = [1, 2]
    accuracy, f1 = training_pipeline.get_model_perfomance()
    assert accuracy == 1.0
    assert f1 == 1.0

@patch('data_ml_assignment.training.train_pipeline.plt.show')
@patch('data_ml_assignment.training.train_pipeline.PlotUtils.plot_confusion_matrix')
def test_render_confusion_matrix(mock_plot, mock_show, training_pipeline):
    training_pipeline.vectorizer = MagicMock()
    training_pipeline.vectorizer.transform.return_value = [[1, 2, 3], [4, 5, 6]]  # Mock transformed data
    training_pipeline.model = MagicMock()
    training_pipeline.model.predict.return_value = [1, 2]
    training_pipeline.y_test = [1, 2]
    training_pipeline.render_confusion_matrix()
    mock_plot.assert_called_once()