import numpy as np
from joblib import dump, load
from .estimator_interface import EstimatorInterface

class BaseModel(EstimatorInterface):
    def __init__(self, model: object = None):
        self.model = model

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> object:
        return self.model.fit(x_train, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        return self.model.predict(x_test)

    def load(self, model_path: str):
        self.model = load(model_path)

    def save(self, path: str = "model.joblib"):
        dump(self.model, path)