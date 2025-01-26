import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from Component.eda_component import EDAComponent

@patch('Component.eda_component.st')
@patch('Component.eda_component.pd.read_csv')  
def test_render_eda(mock_read_csv, mock_streamlit):
    # Mock the return value of `pd.read_csv`
    mock_read_csv.return_value = pd.DataFrame({
        'resume': ['sample resume 1', 'sample resume 2'],
        'label': [1, 2]
    })

    # Initialize the EDAComponent and call the render method
    EDAComp = EDAComponent()
    EDAComp.render()

    # Assertions to verify pandas read_csv was called
    mock_read_csv.assert_called_once()

    # Assertions to verify Streamlit components were called
    mock_streamlit.header.assert_called_once_with("Exploratory Data Analysis")
    mock_streamlit.success.assert_called_once_with("Dataset loaded successfully.")
    assert not mock_streamlit.error.called, "st.error should not be called in this case."


