import pytest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
from eda import render_eda

@patch('eda.pd.read_csv')
def test_render_eda(mock_read_csv):
    mock_read_csv.return_value = pd.DataFrame({
        'resume': ['sample resume 1', 'sample resume 2'],
        'label': [1, 2]
    })
    render_eda()
    # Add assertions to check if Streamlit components are called correctly
    # This is a bit tricky since Streamlit doesn't provide a straightforward way to test UI components
    # You can use `st.write` or `st.bar_chart` to check if they are called with the correct data