from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

class TrainingPipeline:
    def __init__(self, model, vectorizer):
        """
        Initialise le pipeline d'entraînement.
        
        :param model: Modèle de classification (ex: SVC, Naive Bayes).
        :param vectorizer: Méthode de vectorisation (ex: TfidfVectorizer).
        """
        self.pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', model)
        ])

    def train(self, X_train, y_train):
        """
        Entraîne le modèle sur les données d'entraînement.
        
        :param X_train: Données d'entraînement (texte).
        :param y_train: Étiquettes d'entraînement.
        """
        self.pipeline.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """
        Évalue le modèle sur les données de test.
        
        :param X_test: Données de test (texte).
        :param y_test: Étiquettes de test.
        :return: Rapport de classification et accuracy.
        """
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def save_model(self, path):
        """
        Sauvegarde le modèle entraîné.
        
        :param path: Chemin où sauvegarder le modèle.
        """
        joblib.dump(self.pipeline, path)

    def load_model(self, path):
        """
        Charge un modèle pré-entraîné.
        
        :param path: Chemin du modèle à charger.
        """
        self.pipeline = joblib.load(path)