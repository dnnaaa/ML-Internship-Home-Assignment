from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import Dict
import logging

from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.models.database import SessionLocal, PredictionResult
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
from data_ml_assignment.constants import NAIVE_BAYES_PIPELINE_PATH

model = NaiveBayesModel()
model.load(NAIVE_BAYES_PIPELINE_PATH)

inference_router = APIRouter()

# Dependency to get database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@inference_router.post("/inference")
async def inference(request: Dict[str, str], db: Session = Depends(get_db)):
    try:
        resume_text = request.get("text")
        if not resume_text:
            return {"error": "No text provided"}
        
        # Get prediction using the model
        prediction = model.predict([resume_text])[0]
        
        # Save prediction to database
        new_prediction = PredictionResult(
            resume_text=resume_text,
            prediction=str(prediction)
        )
        db.add(new_prediction)
        db.commit()
        
        logging.info(f"Prediction saved successfully: {prediction}")
        return prediction
        
    except Exception as e:
        logging.error(f"Error in inference: {e}")
        db.rollback()
        raise