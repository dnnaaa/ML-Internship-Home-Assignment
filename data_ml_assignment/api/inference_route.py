from fastapi import APIRouter, HTTPException
import logging
from data_ml_assignment.api.schemas import Resume
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
from data_ml_assignment.constants import NAIVE_BAYES_PIPELINE_PATH

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chargement du modèle
model = NaiveBayesModel()
try:
    model.load(NAIVE_BAYES_PIPELINE_PATH)
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle : {str(e)}")
    raise RuntimeError("Impossible de charger le modèle.")

# Création du routeur
inference_router = APIRouter()

@inference_router.post("/inference", response_model=int, description="Prédit la catégorie d'un CV en fonction de son texte.")
def run_inference(resume: Resume):
    """
    Exemple de requête :
    ```json
    {
        "text": "Expérience en développement web avec Python et Django..."
    }
    """
    if not resume.text:
        raise HTTPException(status_code=400, detail="Le texte du CV est vide.")
    
    logger.info(f"Requête reçue pour l'inférence : {resume.text[:50]}...")  # Log les 50 premiers caractères
    
    try:
        prediction = model.predict([resume.text])
        logger.info(f"Prédiction réussie : {prediction}")
        return prediction.tolist()[0]
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {str(e)}")