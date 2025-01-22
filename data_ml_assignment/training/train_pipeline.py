import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
from sklearn.model_selection import train_test_split
from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    REPORTS_PATH,
    LABELS_MAP,
)
from data_ml_assignment.models.tfidf_xgboost import tfidfXGBoost
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

    def train(self, serialize: bool = True, model_name: str = "embedding_model"):
        self.model = tfidfXGBoost()
        self.model.fit(self.x_train, self.y_train)
        if serialize:
            model_path = MODELS_PATH / f"{model_name}.joblib"
            self.model.save(model_path)

    def get_model_performance(self) -> tuple:
        predictions = self.model.predict(self.x_test)
        accuracy = accuracy_score(self.y_test, predictions)
        f1 = f1_score(self.y_test, predictions, average="weighted")
        precision = precision_score(self.y_test, predictions, average="weighted")
        recall = recall_score(self.y_test, predictions, average="weighted")
        return accuracy, f1, precision, recall

    def render_confusion_matrix(self, plot_name: str = "cm_plot"):
        predictions = self.model.predict(self.x_test)
        cm = confusion_matrix(self.y_test, predictions)
        plt.rcParams["figure.figsize"] = (14, 10)
        PlotUtils.plot_confusion_matrix(cm, classes=list(LABELS_MAP.values()), title="Confusion Matrix")
        plot_path = REPORTS_PATH / f"{plot_name}.png"
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close()


    def get_classification_report(self):
        predictions = self.model.predict(self.x_test)
        report = classification_report(self.y_test, predictions, output_dict=True)
        return pd.DataFrame(report).transpose()


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    accuracy, f1, precision, recall, roc_auc = tp.get_model_performance()
    tp.render_confusion_matrix()
    tp.render_roc_curve()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1}, PRECISION = {precision}, RECALL = {recall}, ROC-AUC = {roc_auc}")