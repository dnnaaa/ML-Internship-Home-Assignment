import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from dashboard import EDAComponent, InferenceComponent, main

@pytest.fixture
def mock_resumes_df():
    """Fixture to provide a mock dataset."""
    return pd.DataFrame({
        "resume": ["Resume text 1", "Resume text 2", "Resume text 3"],
        "Label": ["Java Developer", "Data Scientist", "Python Developer"]
    })

@patch("dashboard.pd.read_csv")
def test_eda_component_render(mock_read_csv, mock_resumes_df, mocker):
    """Test the EDAComponent render method."""
    # Mock pandas read_csv to return a test dataframe
    mock_read_csv.return_value = mock_resumes_df

    # Mock Streamlit functions
    mock_st = mocker.patch("streamlit", autospec=True)

    # Render the EDA component
    eda = EDAComponent()
    eda.render()

    # Assertions
    mock_read_csv.assert_called_once()  # Check if read_csv was called
    mock_st.header.assert_called_with("Exploratory Data Analysis")  # Check header
    mock_st.dataframe.assert_called_with(mock_resumes_df.head(10))  # Check dataframe preview

@patch("dashboard.load")
@patch("dashboard.add_prediction_to_db")
@patch("dashboard.get_all_predictions")
def test_inference_component_render(mock_get_predictions, mock_add_to_db, mock_load, mocker):
    """Test the InferenceComponent render method."""
    # Mock the loaded model and vectorizer
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = "vectorized_text"
    mock_model.predict.return_value = ["Python Developer"]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    mock_load.side_effect = [mock_model, mock_vectorizer]

    # Mock Streamlit
    mock_st = mocker.patch("streamlit", autospec=True)
    mock_st.text_area.return_value = "Sample resume text"

    # Mock database calls
    mock_get_predictions.return_value = []

    # Render the inference component
    inference = InferenceComponent()
    inference.render()

    # Assertions
    mock_load.assert_called()  # Check if models were loaded
    mock_st.text_area.assert_called_with("Enter resume text:", height=300)  # Check input
    mock_model.predict.assert_called_with(["vectorized_text"])  # Check prediction
    mock_model.predict_proba.assert_called_once()  # Check probabilities
    mock_add_to_db.assert_called_with("Sample resume text", "Python Developer")  # DB insertion
    mock_st.success.assert_any_call("Prediction saved to database.")  # Success message

@patch("dashboard.EDAComponent.render")
@patch("dashboard.InferenceComponent.render")
def test_main(mock_inference_render, mock_eda_render, mocker):
    """Test the main function to ensure proper navigation."""
    # Mock Streamlit
    mock_st = mocker.patch("streamlit", autospec=True)
    mock_st.sidebar.radio.return_value = "EDA"  # Simulate user selecting "EDA"

    # Call main
    main()

    # Assertions
    mock_eda_render.assert_called_once()  # EDA should render
    mock_inference_render.assert_not_called()  # Inference should not render

    # Simulate user selecting "Inference"
    mock_st.sidebar.radio.return_value = "Inference"
    main()

    # Assertions
    mock_inference_render.assert_called_once()  # Inference should render

