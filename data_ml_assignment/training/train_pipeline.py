import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP,
)
from data_ml_assignment.utils.plot_utils import PlotUtils
import joblib

class TrainingPipeline:
    def __init__(self):
        df = pd.read_csv(RAW_DATASET_PATH)

        text = df["resume"]
        y = df["label"]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text, y, test_size=0.2, random_state=0
        )

        self.model = None

    def train(self, serialize: bool = True, model_name: str = "advanced_model"):
        # Define pipeline with TF-IDF vectorizer and Logistic Regression
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', LogisticRegression(max_iter=1000, random_state=0)),
        ])

        # Hyperparameter tuning
        param_grid = {
            'clf__C': [0.1, 1, 10],
            'clf__solver': ['liblinear', 'lbfgs']
        }
        grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1_weighted')
        grid_search.fit(self.x_train, self.y_train)

        self.model = grid_search.best_estimator_

        if serialize:
            model_path = MODELS_PATH / f"{model_name}.joblib"
            joblib.dump(self.model, model_path)

    def get_model_perfomance(self) -> tuple:
        predictions = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, predictions), f1_score(
            self.y_test, predictions, average="weighted"
        )

    def render_confusion_matrix(self):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        # Generate the confusion matrix plot
        PlotUtils.plot_confusion_matrix(
            cm, classes=list(LABELS_MAP.values()), title="Advanced Model"
        )

        # Display the confusion matrix plot in Streamlit
        st.pyplot(plt)
        plt.close()  # Close the plot to avoid overlapping plots in Streamlit
