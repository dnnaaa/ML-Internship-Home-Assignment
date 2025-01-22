from sklearn.ensemble import RandomForestClassifier
from data_ml_assignment.models.base_model import BaseModel
from data_ml_assignment.models.estimator_interface import EstimatorInterface

class RandomForestModel(BaseModel, EstimatorInterface):
    def __init__(self):
        super().__init__(
            vectorizer_type="tfidf",
            classifier=RandomForestClassifier(n_estimators=100),
            vectorizer_params={"max_features": 5000, "stop_words": "english"}
        )

    def get_config(self):
        return {
            "model_type": "random_forest",
            "vectorizer": self.vectorizer_type,
            "classifier": "RandomForest"
        }