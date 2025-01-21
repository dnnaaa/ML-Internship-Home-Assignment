
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib  

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

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text, y, test_size=0.2, random_state=0
        )

        self.model = None
        self.vectorizer = None

    def train(self, serialize: bool = True, model_name: str = "model"):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        x_train_vec = self.vectorizer.fit_transform(self.x_train)
        x_test_vec = self.vectorizer.transform(self.x_test)

        self.model = LogisticRegression()

        param_grid = {
            "C": [0.1, 1, 10],
            "penalty": ["l2"],  # Use l2 penalty
            "solver": ["lbfgs"],  # Use lbfgs solver
        }
        grid_search = GridSearchCV(self.model, param_grid, cv=3, scoring="f1_weighted")
        grid_search.fit(x_train_vec, self.y_train)

        self.model = grid_search.best_estimator_

        if serialize:
            model_path = MODELS_PATH / f"{model_name}.joblib"
            joblib.dump(
                {"model": self.model, "vectorizer": self.vectorizer}, model_path
            )

    def get_model_perfomance(self) -> tuple:
        x_test_vec = self.vectorizer.transform(self.x_test)
        predictions = self.model.predict(x_test_vec)
        return accuracy_score(self.y_test, predictions), f1_score(
            self.y_test, predictions, average="weighted"
        )

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        x_test_vec = self.vectorizer.transform(self.x_test)
        predictions = self.model.predict(x_test_vec)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        PlotUtils.plot_confusion_matrix(
            cm, classes=list(LABELS_MAP.values()), title="Confusion Matrix"
        )

        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    accuracy, f1_score = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1_score}")


