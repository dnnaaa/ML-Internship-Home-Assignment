import pytest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
import io

from data_ml_assignment.dashboard.components.training import TrainingComponent
from data_ml_assignment.training.train_pipeline import TrainingPipeline

@pytest.fixture
def mock_pipeline():
    """Create mock training pipeline."""
    pipeline = Mock(spec=TrainingPipeline)
    pipeline.accuracy = 0.85
    pipeline.f1 = 0.83
    return pipeline

@pytest.fixture
def training_component():
    """Create training component for testing."""
    return TrainingComponent()

@pytest.fixture
def mock_confusion_matrix():
    """Create mock confusion matrix image."""
    img = Image.new('RGB', (100, 100), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr

def test_handle_training(training_component, mock_pipeline, mock_confusion_matrix):
    """Test training handling process."""
    with patch('data_ml_assignment.training.train_pipeline.TrainingPipeline', return_value=mock_pipeline):
        with patch('PIL.Image.open', return_value=mock_confusion_matrix):
            # Simuler l'entraînement
            training_component._handle_training("test_model", True)
            
            # Vérifier que les méthodes ont été appelées
            mock_pipeline.train.assert_called_once_with(serialize=True, model_name="test_model")
            mock_pipeline.render_confusion_matrix.assert_called_once()
            mock_pipeline.get_model_perfomance.assert_called_once()

def test_display_metrics(training_component):
    """Test metrics display."""
    with patch('streamlit.columns') as mock_columns:
        # Créer des colonnes mock
        col1, col2 = Mock(), Mock()
        mock_columns.return_value = [col1, col2]
        
        # Tester l'affichage des métriques
        training_component._display_metrics(0.85, 0.83)
        
        # Vérifier les appels aux métriques
        col1.metric.assert_called_once_with(
            label="Accuracy score",
            value="0.8500"
        )
        col2.metric.assert_called_once_with(
            label="F1 score",
            value="0.8300"
        ) 