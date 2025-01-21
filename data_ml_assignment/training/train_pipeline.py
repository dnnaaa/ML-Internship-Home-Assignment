import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import joblib  # Import joblib for model serialization

from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP,
)
from data_ml_assignment.utils.plot_utils import PlotUtils


class TrainingPipeline:
    def __init__(self):
        df = pd.read_csv(RAW_DATASET_PATH)
        text = df["resume"]
        y = df["label"]

        # Use TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(text)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=0
        )

        self.model = None

    def train(self, serialize: bool = True, model_name: str = "model"):
        # Use Logistic Regression with hyperparameter tuning
        param_grid = {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"]
        }

        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1_weighted")
        grid_search.fit(self.x_train, self.y_train)

        self.model = grid_search.best_estimator_

        if serialize:
            model_path = MODELS_PATH / f"{model_name}.joblib"
            joblib.dump(self.model, model_path)  # Save the model using joblib

    def get_model_perfomance(self) -> tuple:
        # Test set performance
        test_predictions = self.model.predict(self.x_test)
        test_accuracy = accuracy_score(self.y_test, test_predictions)
        test_f1 = f1_score(self.y_test, test_predictions, average="weighted")

        # Training set performance
        train_predictions = self.model.predict(self.x_train)
        train_accuracy = accuracy_score(self.y_train, train_predictions)
        train_f1 = f1_score(self.y_train, train_predictions, average="weighted")

        return (train_accuracy, train_f1), (test_accuracy, test_f1)

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        PlotUtils.plot_confusion_matrix(
            cm, classes=list(LABELS_MAP.values()), title="Logistic Regression"
        )

        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    (train_accuracy, train_f1), (test_accuracy, test_f1) = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print(f"Training Accuracy = {train_accuracy:.4f}, Training F1 Score = {train_f1:.4f}")
    print(f"Test Accuracy = {test_accuracy:.4f}, Test F1 Score = {test_f1:.4f}")