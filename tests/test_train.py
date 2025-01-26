from unittest.mock import patch, MagicMock
import pytest
from Component.training_component import TrainingComponent

@patch('Component.training_component.st')  # Mock Streamlit
@patch('Component.training_component.display_metrics')  # Mock display_metrics
@patch('Component.training_component.TrainingPipeline')  # Mock TrainingPipeline
def test_render_training(mock_training_pipeline, mock_display_metrics, mock_streamlit):
    # Mock TrainingPipeline methods
    mock_pipeline_instance = MagicMock()
    mock_training_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.get_model_perfomance.return_value = (0.95, 0.90)  

    # Simulate Streamlit interactions
    mock_streamlit.text_input.return_value = "Naive Bayes"
    mock_streamlit.checkbox.return_value = True
    mock_streamlit.button.return_value = True  # Simulate clicking the "Train pipeline" button

    # Initialize and render the TrainingComponent
    training_comp = TrainingComponent()
    training_comp.render()

    
    # Assertions for Streamlit interactions
    mock_streamlit.header.assert_called_once_with("Pipeline Training")
    mock_streamlit.info.assert_called_once()
    mock_streamlit.text_input.assert_called_once_with("Pipeline name", placeholder="Naive Bayes")
    mock_streamlit.checkbox.assert_called_once_with("Save pipeline")
    mock_streamlit.button.assert_called_once_with("Train pipeline")
    mock_streamlit.spinner.assert_called_once_with("Training pipeline, please wait...")

    # Assertions for TrainingPipeline interactions
    mock_training_pipeline.assert_called_once()  # Ensure TrainingPipeline is instantiated
    mock_pipeline_instance.train.assert_called_once_with(serialize=True, model_name="Naive Bayes")
    mock_pipeline_instance.get_model_perfomance.assert_called_once()
    accuracy, f1_score = mock_pipeline_instance.get_model_perfomance.return_value
    mock_display_metrics.assert_called_once_with(accuracy, f1_score)
    mock_pipeline_instance.render_confusion_matrix.assert_called_once()

    # Ensure no errors were logged
    mock_streamlit.error.assert_not_called()

# Test for failure case
@patch('Component.training_component.st')  # Mock Streamlit
@patch('Component.training_component.display_metrics')  # Mock display_metrics
@patch('Component.training_component.TrainingPipeline')  # Mock TrainingPipeline
def test_render_training_failure(mock_training_pipeline, mock_display_metrics, mock_streamlit):
    # Mock TrainingPipeline methods to raise an exception
    mock_pipeline_instance = MagicMock()
    mock_training_pipeline.return_value = mock_pipeline_instance
    mock_pipeline_instance.train.side_effect = Exception("Training error!")

    # Simulate Streamlit interactions
    mock_streamlit.text_input.return_value = "Naive Bayes"
    mock_streamlit.checkbox.return_value = True
    mock_streamlit.button.return_value = True  # Simulate clicking the "Train pipeline" button

    # Initialize and render the TrainingComponent
    training_comp = TrainingComponent()
    training_comp.render()

    # Assertions for error handling
    mock_streamlit.error.assert_called_once_with("Failed to train the pipeline!")
    mock_streamlit.exception.assert_called_once()
    mock_pipeline_instance.train.assert_called_once_with(serialize=True, model_name="Naive Bayes")
    mock_display_metrics.assert_not_called()  # Ensure metrics are not displayed on failure
    mock_pipeline_instance.render_confusion_matrix.assert_not_called()
