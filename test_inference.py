import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dashboard import Prediction, SessionLocal

# Mock the database session
@pytest.fixture
def mock_db_session():
    engine = create_engine("sqlite:///:memory:")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Prediction.metadata.create_all(engine)
    return SessionLocal()

# Test saving prediction to the database
def test_save_prediction(mock_db_session):
    with patch("dashboard.SessionLocal", return_value=mock_db_session):
        # Create a mock prediction
        prediction = Prediction(
            resume_name="Java Developer",
            prediction_result="Java Developer",
            confidence_score=5  
        )
        mock_db_session.add(prediction)
        mock_db_session.commit()

        # Verify the prediction was saved
        saved_prediction = mock_db_session.query(Prediction).first()
        assert saved_prediction.resume_name == "Java Developer"
        assert saved_prediction.prediction_result == "Java Developer"
        assert saved_prediction.confidence_score == 5  

# Test the inference API call
def test_inference_api_call():
    with patch("requests.post") as mock_post:
        mock_post.return_value.text = "1.0"  
        response = mock_post("http://localhost:9000/api/inference", json={"text": "sample resume text"})
        
        assert response.text == "1.0"