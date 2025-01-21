from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict
import logging
import numpy as np

from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.models.database import SessionLocal, PredictionResult
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
from data_ml_assignment.constants import NAIVE_BAYES_PIPELINE_PATH


model = NaiveBayesModel()
model.load(NAIVE_BAYES_PIPELINE_PATH)

inference_router = APIRouter()

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
        
        # Convert numpy.int64 to native Python int
        if isinstance(prediction, np.integer):
            prediction = int(prediction)
        
        # Save to database
        db_prediction = PredictionResult(resume_text=resume_text, prediction=str(prediction))
        db.add(db_prediction)
        db.commit()
        db.refresh(db_prediction)
        
        # Query all predictions from the database
        predictions = db.query(PredictionResult).all()
        
        return {
            "resume_text": resume_text,
            "prediction": prediction,
            "all_predictions": [{"resume_text": p.resume_text, "prediction": p.prediction} for p in predictions]
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))