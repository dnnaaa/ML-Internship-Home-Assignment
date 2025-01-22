import pytest
from unittest.mock import patch, MagicMock, mock_open
import streamlit as st
from inference import render_inference, Prediction

@patch('inference.requests.post')
@patch('inference.session.add')
@patch('inference.st.button')  # Mock the Streamlit button
@patch('inference.st.selectbox')  # Mock the Streamlit selectbox
@patch('builtins.open', new_callable=mock_open, read_data="sample resume text")  # Mock file reading
def test_render_inference(mock_file, mock_selectbox, mock_button, mock_session_add, mock_post):
    # Mock the selectbox to return a sample resume
    mock_selectbox.return_value = "Software Engineer"

    # Mock the button to return True (simulate button click)
    mock_button.return_value = True

    # Mock the API response
    mock_post.return_value = MagicMock()
    mock_post.return_value.text = "1"  # Simulate the API returning label "1"

    # Call the function
    render_inference()

    # Assert that the file was opened
    mock_file.assert_called_once()

    # Assert that the API was called
    mock_post.assert_called_once_with(
        "http://localhost:9000/api/inference",
        json={"text": "sample resume text"}  # Adjust this to match the expected input
    )

    # Assert that the prediction was saved to the database
    mock_session_add.assert_called_once()

def test_prediction_model():
    prediction = Prediction(resume_text="sample resume", predicted_label="Software Engineer")
    assert prediction.resume_text == "sample resume"
    assert prediction.predicted_label == "Software Engineer"