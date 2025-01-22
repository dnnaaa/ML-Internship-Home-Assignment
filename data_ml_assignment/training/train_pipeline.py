import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from pathlib import Path

from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP
)
from data_ml_assignment.models import (
    NaiveBayesModel,
    SVCModel,
    XGBoostModel,
    LogisticRegressionModel
)
from data_ml_assignment.utils.plot_utils import PlotUtils

class TrainingPipeline:
    # Class-level model registry (MUST be defined first)
    MODEL_REGISTRY = {
        "naive_bayes": NaiveBayesModel,
        "svc": SVCModel,
        "xgboost": XGBoostModel,
        "logistic_regression": LogisticRegressionModel
    }

    def __init__(self, model_type: str = "naive_bayes"):
        # Load data
        df = pd.read_csv(RAW_DATASET_PATH)
        text = df["resume"]
        y = df["label"]

        # Split data
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text, y, test_size=0.2, random_state=0
        )
        
        # Initialize model
        self.model_type = model_type
        self.model = self.MODEL_REGISTRY[model_type]()

    def train(self, serialize: bool = True, model_name: str = None):
        """Train and optionally save the model"""
        self.model.fit(self.x_train, self.y_train)
        
        if serialize:
            model_name = model_name or f"{self.model_type}_pipeline"
            model_path = MODELS_PATH / f"{model_name}.joblib"
            self.model.save(model_path)

    def get_model_perfomance(self) -> tuple:
        """Return accuracy and F1 score"""
        predictions = self.model.predict(self.x_test)
        return (
            accuracy_score(self.y_test, predictions),
            f1_score(self.y_test, predictions, average="weighted")
        )

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        """Generate confusion matrix plot"""
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        
        plt.rcParams["figure.figsize"] = (14, 10)
        plt.clf()
        
        PlotUtils.plot_confusion_matrix(
            cm,
            classes=list(LABELS_MAP.values()),
            title=f"{self.model_type.replace('_', ' ').title()} Confusion Matrix"
        )
        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()