from fastapi import APIRouter
from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.models.LinearRegressionModel import LogisticRegressionModel
from data_ml_assignment.constants import LR_PIPELINE_PATH

model = LogisticRegressionModel()
model.load(LR_PIPELINE_PATH)

inference_router = APIRouter()

@inference_router.post("/inference")
def run_inference(resume: Resume):
    # Get the predicted label and confidence
    prediction = model.predict([resume.text])  # Predict the label
    # Return a JSON response with label and confidence
    return prediction.tolist()[0]
