import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore


from sklearn.model_selection import train_test_split, cross_val_score, learning_curve # type: ignore
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix # type: ignore
from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP,
)
from data_ml_assignment.models.naive_bayes_model import NaiveBayesModel
from data_ml_assignment.utils.plot_utils import PlotUtils
from data_ml_assignment.models.svm_model import SVMModel
from data_ml_assignment.models.RandomForest_model import RandomForestModel
from data_ml_assignment.models.LogisticRegression_model import LogisticRegressionModel 
from data_ml_assignment.models.xgbc_model import XGBCModel 
from data_ml_assignment.models.svc_model import SVCModel 


class TrainingPipeline:
    def __init__(self):
        df = pd.read_csv(RAW_DATASET_PATH)

        text = df["resume"]
        y = df["label"]

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            text, y, test_size=0.2, random_state=0
        )

        self.model = None

    def train(self, serialize: bool = True, model_name: str = "model"):
        self.model = SVCModel()
        self.model.fit(self.x_train, self.y_train)

        model_path = MODELS_PATH / f"{model_name}.joblib"
        if serialize:
            self.model.save(model_path)

    def get_model_perfomance(self) -> dict:
            predictions_test = self.model.predict(self.x_test)
            predictions_train = self.model.predict(self.x_train)

            train_accuracy = accuracy_score(self.y_train, predictions_train)
            test_accuracy = accuracy_score(self.y_test, predictions_test)
            test_f1 = f1_score(self.y_test, predictions_test, average="weighted")

            return {
                train_accuracy,
                test_accuracy,
                test_f1,
            }

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)

        PlotUtils.plot_confusion_matrix(
            cm, classes=list(LABELS_MAP.values()), title="SVCModel"
        )

        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.show()




if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True, model_name="SVCModel")

    accuracy, f1_score = tp.get_model_perfomance()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1_score}")

    
