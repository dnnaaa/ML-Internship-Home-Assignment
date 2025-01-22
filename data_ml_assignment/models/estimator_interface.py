import numpy as np

class EstimatorInterface:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> object:
        raise NotImplementedError
    
    def predict(self, x_test: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        raise NotImplementedError
    
    def load(self, model_path: str) -> None:
        raise NotImplementedError