import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import json
from Component.inference_component import InferenceComponent


@patch('Component.inference_component.st')  # Patch Streamlit
@patch('Component.inference_component.load_sample_text')  # Mock load_sample_text
@patch('Component.inference_component.run_inference')  # Mock run_inference
@patch('Component.inference_component.save_inference')  # Mock save_inference
@patch('Component.inference_component.show_inference')  # Mock show_inference
@patch('Component.inference_component.delete_inference')  # Mock delete_inference
@patch('Component.inference_component.LABELS_MAP')  # Mock LABELS_MAP
def test_render_inference(mock_labels_map, mock_delete_inference, mock_show_inference, mock_save_inference, mock_run_inference, mock_load_sample_text, mock_streamlit):
    # Mock LABELS_MAP values
    mock_labels_map.values.return_value = ["Label 1", "Label 2", "Label 3"]
    mock_labels_map.get.return_value = "Label 1"  # Mock the return value for LABELS_MAP.get()

    # Mocking the return values of the helpers
    mock_load_sample_text.return_value = "This is a sample resume text."
    mock_run_inference.return_value = "1"
    mock_save_inference.return_value = None
    mock_show_inference.return_value = json.dumps({
        "predictions": [
            {"id": 1, "text": "Sample resume text 1", "label": "Label 1"},
            {"id": 2, "text": "Sample resume text 2", "label": "Label 2"}
        ]
    })
    mock_delete_inference.return_value = "Inference deleted successfully."

    # Simulate Streamlit interactions
    mock_streamlit.selectbox.return_value = "Label 1"
    mock_streamlit.button.side_effect = [True, False]  # Simulate clicking "Run Inference" and not clicking "Delete inference"

    # Initialize the InferenceComponent and call render
    inference_comp = InferenceComponent()
    inference_comp.render()

    # Assertions for Streamlit interactions
    mock_streamlit.header.assert_called_once_with("Resume Inference")
    mock_streamlit.info.assert_called_once()
    mock_streamlit.selectbox.assert_called_once_with(
        "Resume samples for inference",
        options=("Label 1", "Label 2", "Label 3"),
        index=0
    )
    mock_streamlit.spinner.assert_called_once_with("Running inference...")

    # Assertions for helper functions
    mock_load_sample_text.assert_called_once_with("Label 1")
    mock_run_inference.assert_called_once_with("This is a sample resume text.")
    mock_save_inference.assert_called_once_with("This is a sample resume text.", "Label 1")
    mock_show_inference.assert_called_once()
    mock_delete_inference.assert_not_called()  # "Delete inference" button was not clicked

    # Assertions for displaying results
    mock_streamlit.success.assert_called_once_with("Done!")
    mock_streamlit.metric.assert_called_once_with(label="Status", value="Resume label: Label 1")
    mock_streamlit.dataframe.assert_called_once()
