from fastapi import APIRouter

from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
from data_ml_assignment.constants import NAIVE_BAYES_PIPELINE_PATH


model = NaiveBayesModel()
model.load(SVM_PIPELINE_PATH )

inference_router = APIRouter()


@inference_router.post("/inference")
def run_inference(resume: Resume):
    prediction = model.predict([resume.text])
    return prediction.tolist()[0]
