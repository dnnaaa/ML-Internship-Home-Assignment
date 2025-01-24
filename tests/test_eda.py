import pytest
import pandas as pd
from unittest.mock import patch, mock_open
import plotly.graph_objects as go

from data_ml_assignment.dashboard.components.eda import EDAComponent
from data_ml_assignment.constants import LABELS_MAP

@pytest.fixture
def mock_data():
    """Create mock DataFrame for testing."""
    return pd.DataFrame({
        'resume': ['python developer experience', 'java developer skills'],
        'label': [0, 1],
        'label_name': ['Python Developer', 'Java Developer'],
        'word_count': [3, 3],
        'char_count': [27, 22]
    })

@pytest.fixture
def eda_component(mock_data):
    """Create EDA component with mocked data."""
    with patch('pandas.read_csv', return_value=mock_data):
        return EDAComponent()

def test_preprocess_text(eda_component):
    """Test text preprocessing."""
    text = "Python3 & Java! Skills-2023"
    processed = eda_component.preprocess_text(text)
    assert processed == "python java skills"

def test_get_word_frequencies(eda_component):
    """Test word frequency calculation."""
    text = "python java python skills"
    frequencies = eda_component.get_word_frequencies([text])
    assert frequencies['python'] == 2
    assert frequencies['java'] == 1
    assert frequencies['skills'] == 1

def test_create_distribution_plot(eda_component):
    """Test distribution plot creation."""
    plot = eda_component.create_distribution_plot('word_count', 'Word Count Distribution')
    assert isinstance(plot, go.Figure)
    assert plot.layout.title.text == 'Word Count Distribution'

def test_create_label_distribution(eda_component):
    """Test label distribution plot creation."""
    plot = eda_component.create_label_distribution()
    assert isinstance(plot, go.Figure)
    assert plot.layout.title.text == 'Resume Label Distribution' 