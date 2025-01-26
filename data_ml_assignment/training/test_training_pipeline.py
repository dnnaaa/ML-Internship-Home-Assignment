import pytest
from unittest.mock import patch, MagicMock
from sklearn.metrics import accuracy_score, f1_score
from data_ml_assignment.training_pipeline import TrainingPipeline
from data_ml_assignment.constants import RAW_DATASET_PATH


@pytest.fixture
def mock_dataframe():
    """Mock dataset."""
    import pandas as pd
    return pd.DataFrame({
        "resume": ["This is a sample resume text", "Another resume text", "More data"],
        "label": ["Python Developer", "Data Scientist", "Java Developer"]
    })


@patch("data_ml_assignment.training_pipeline.pd.read_csv")
@patch("data_ml_assignment.training_pipeline.dump")
def test_training_pipeline_initialization(mock_dump, mock_read_csv, mock_dataframe):
    """Test the initialization of the TrainingPipeline class."""
    mock_read_csv.return_value = mock_dataframe

    # Initialize pipeline
    tp = TrainingPipeline()

    # Assertions
    mock_read_csv.assert_called_once_with(RAW_DATASET_PATH)
    assert tp.x_train.shape[0] > 0
    assert tp.x_test.shape[0] > 0
    assert hasattr(tp, "model")
    assert hasattr(tp, "vectorizer")


@patch("data_ml_assignment.training_pipeline.dump")
@patch.object(TrainingPipeline, "get_model_perfomance")
def test_training_pipeline_train(mock_get_model_perfomance, mock_dump, mock_dataframe):
    """Test the train method."""
    # Mocking data and performance
    mock_get_model_perfomance.return_value = (0.9, 0.85)

    # Initialize pipeline and train
    tp = TrainingPipeline()
    tp.train(serialize=True, model_name="test_model")

    # Assertions
    mock_dump.assert_called()  # Ensure model and vectorizer are serialized
    assert mock_get_model_perfomance.call_count == 0  # Performance isn't directly called here


@patch.object(TrainingPipeline, "render_confusion_matrix")
@patch("data_ml_assignment.training_pipeline.accuracy_score", return_value=0.9)
@patch("data_ml_assignment.training_pipeline.f1_score", return_value=0.85)
def test_get_model_performance(mock_f1_score, mock_accuracy_score, mock_render_cm, mock_dataframe):
    """Test the get_model_perfomance method."""
    # Initialize pipeline
    tp = TrainingPipeline()

    # Call performance method
    accuracy, f1 = tp.get_model_perfomance()

    # Assertions
    mock_accuracy_score.assert_called_once()
    mock_f1_score.assert_called_once()
    assert accuracy == 0.9
    assert f1 == 0.85


@patch("data_ml_assignment.training_pipeline.PlotUtils.plot_confusion_matrix")
@patch("data_ml_assignment.training_pipeline.plt.savefig")
@patch("data_ml_assignment.training_pipeline.plt.show")
def test_render_confusion_matrix(mock_show, mock_savefig, mock_plot_cm, mock_dataframe):
    """Test the render_confusion_matrix method."""
    # Initialize pipeline
    tp = TrainingPipeline()

    # Call confusion matrix rendering
    tp.render_confusion_matrix(plot_name="test_cm_plot")

    # Assertions
    mock_plot_cm.assert_called_once()
    mock_savefig.assert_called_once()
    mock_show.assert_called_once()
