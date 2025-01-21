from fastapi import APIRouter
from data_ml_assignment.api.schemas import Resume , PredictionResponse 
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel 
from data_ml_assignment.models.logistic_model import LogisticModel
from data_ml_assignment.constants import NAIVE_BAYES_PIPELINE_PATH
from data_ml_assignment.constants import LOGISTIC_MODEL_PIPELINE_PATH
from .database import save_prediction_to_db, get_all_predictions
from fastapi import FastAPI, HTTPException

model = LogisticModel()
model.load(LOGISTIC_MODEL_PIPELINE_PATH)
inference_router = APIRouter()


#EndPoint to run the inferce or the predict function
@inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]

#EndPoint to save the inference on the database
@inference_router.post("/save")
def save_inference(prediction: PredictionResponse):
    save_prediction_to_db(prediction.text, prediction.predict)

#EndPoint to show all the inferences from the database
@inference_router.post("/get")
def get_inference():
    predictions = get_all_predictions()
    return {"predictions": predictions}