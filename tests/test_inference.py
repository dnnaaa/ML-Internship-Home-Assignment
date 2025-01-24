import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import pandas as pd

from data_ml_assignment.dashboard.components.inference import InferenceComponent
from data_ml_assignment.models.prediction import Session, Prediction

@pytest.fixture
def mock_session():
    """Create a mock database session."""
    session = Mock(spec=Session)
    return session

@pytest.fixture
def inference_component(mock_session):
    """Create an inference component with mocked session."""
    with patch('data_ml_assignment.dashboard.components.inference.Session') as MockSession:
        MockSession.return_value = mock_session
        component = InferenceComponent()
        return component, mock_session

def test_save_prediction(inference_component):
    """Test saving a prediction to the database."""
    component, mock_session = inference_component
    
    # Appeler la méthode de sauvegarde
    component._save_prediction("Software Engineer", "Label 1")
    
    # Vérifier que add et commit ont été appelés
    assert mock_session.add.called
    assert mock_session.commit.called
    
    # Vérifier les données de la prédiction
    prediction = mock_session.add.call_args[0][0]
    assert prediction.resume_type == "Software Engineer"
    assert prediction.predicted_label == "Label 1"

@pytest.fixture
def mock_api_response():
    """Create a mock API response."""
    mock_response = Mock()
    mock_response.text = "1"  # Simuler une réponse de l'API
    return mock_response

def test_handle_inference(inference_component, mock_api_response):
    """Test the inference handling process."""
    component, mock_session = inference_component
    
    with patch('requests.post', return_value=mock_api_response):
        with patch.object(component, '_read_sample_file', return_value="Sample resume text"):
            # Simuler une inférence
            component._handle_inference("Software Engineer")
            
            # Vérifier que la prédiction a été sauvegardée
            assert mock_session.add.called
            assert mock_session.commit.called

def test_read_sample_file(inference_component):
    """Test reading a sample file."""
    component, _ = inference_component
    
    with patch('builtins.open', mock_open(read_data="Sample resume content")):
        content = component._read_sample_file("Software Engineer")
        assert content == "Sample resume content"

def test_call_inference_api(inference_component, mock_api_response):
    """Test calling the inference API."""
    component, _ = inference_component
    
    with patch('requests.post', return_value=mock_api_response) as mock_post:
        response = component._call_inference_api("Sample text")
        
        # Vérifier que l'API a été appelée avec les bons paramètres
        mock_post.assert_called_with(
            "http://localhost:9000/api/inference",
            json={"text": "Sample text"}
        )
        assert response.text == "1" 