from fastapi import APIRouter
import joblib
from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.constants import SVM_PIPELINE_PATH, VECTORIZER_PATH

# Chargement du modèle SVM et du vectorizer
model = joblib.load(SVM_PIPELINE_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)  # Charger le vectorizer utilisé lors de l'entraînement

inference_router = APIRouter()

@inference_router.post("/inference")
def run_inference(resume: Resume):
    # Prétraiter le texte d'entrée en utilisant le même vectorizer que celui utilisé pour l'entraînement
    resume_vectorized = vectorizer.transform([resume.text])
    
    # Prédire la classe avec le modèle SVM
    prediction = model.predict(resume_vectorized)
    
    # Retourner la prédiction sous forme de liste
    return prediction.tolist()[0]
