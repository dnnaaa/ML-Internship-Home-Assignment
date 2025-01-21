import logging
logging.basicConfig(level=logging.INFO)

from fastapi import FastAPI
from data_ml_assignment.api.inference_route import inference_router
from data_ml_assignment.api.constants import APP_NAME, API_PREFIX
from data_ml_assignment.models.database import SessionLocal, PredictionResult

def server() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    
    app.include_router(inference_router, prefix=API_PREFIX)

    @app.get("/")
    def read_root():
        return {"message": f"Welcome to {APP_NAME} API"}

    @app.get("/api/predictions")
    def get_predictions():
        session = SessionLocal()
        try:
            predictions = session.query(PredictionResult).all()
            logging.info(f"Fetched {len(predictions)} predictions.")
            return [
                {"resume_text": pred.resume_text, "prediction": pred.prediction}
                for pred in predictions
            ]
        finally:
            session.close()

    return app