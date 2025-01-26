import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from joblib import dump

from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP,
)
from data_ml_assignment.utils.plot_utils import PlotUtils


class TrainingPipeline:
    def __init__(self):
        # Charger les données
        df = pd.read_csv(RAW_DATASET_PATH)

        text = df["resume"]
        y = df["label"]

        # Séparer les données en ensembles d'entraînement et de test
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text, y, test_size=0.2, random_state=0
        )

        # Initialisation du vectoriseur TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        self.x_train = self.vectorizer.fit_transform(self.x_train)
        self.x_test = self.vectorizer.transform(self.x_test)

        # Initialisation du modèle
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def train(self, serialize: bool = True, model_name: str = "improved_model"):
        # Entraîner le modèle
        self.model.fit(self.x_train, self.y_train)

        # Sauvegarder le modèle et le vectoriseur si nécessaire
        model_path = MODELS_PATH / f"{model_name}.joblib"
        vectorizer_path = MODELS_PATH / f"{model_name}_vectorizer.joblib"
        if serialize:
            dump(self.model, model_path)
            dump(self.vectorizer, vectorizer_path)
            print(f"Model saved to {model_path}")
            print(f"Vectorizer saved to {vectorizer_path}")

    def get_model_perfomance(self) -> tuple:
        # Prédictions et métriques
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions), f1_score(
            self.y_test, predictions, average="weighted"
        )

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        # Matrice de confusion
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        PlotUtils.plot_confusion_matrix(
            cm, classes=list(LABELS_MAP.values()), title="Random Forest"
        )

        # Sauvegarder et afficher la matrice de confusion
        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    accuracy, f1 = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1}")
