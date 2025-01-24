from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
from typing import Dict, Tuple

from data_ml_assignment.constants import (
    RAW_DATASET_PATH,
    MODELS_PATH,
    CM_PLOT_PATH
)

class TrainingPipeline:
    """Enhanced training pipeline with multiple models."""
    
    MODELS = {
        'SVM': LinearSVC(C=1.0, class_weight='balanced', max_iter=1000, dual=False),
        'Random Forest': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
        'Naive Bayes': MultinomialNB(),
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Decision Tree': DecisionTreeClassifier(class_weight='balanced'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
    }
    
    def __init__(self):
        self.best_pipeline = None
        self.best_model_name = None
        self.label_encoder = LabelEncoder()
        self.model_path = MODELS_PATH
        self.model_scores = {}
        
    def _create_pipeline(self, classifier) -> Pipeline:
        """Create a pipeline with TF-IDF and the specified classifier."""
        return Pipeline([
            ('vectorizer', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                stop_words='english'
            )),
            ('classifier', classifier)
        ])

    def train(self, serialize: bool = False, model_name: str = "best_model", progress_callback=None):
        """Train multiple models and select the best one."""
        # Charger et préparer les données
        df = pd.read_csv(RAW_DATASET_PATH)
        X = df['resume']
        y = self.label_encoder.fit_transform(df['label'])
        
        best_f1 = 0
        best_accuracy = 0
        
        total_models = len(self.MODELS)
        
        # Entraîner tous les modèles
        for idx, (name, classifier) in enumerate(self.MODELS.items(), 1):
            try:
                if progress_callback:
                    progress_callback(f"Training {name} ({idx}/{total_models})...")
                
                pipeline = self._create_pipeline(classifier)
                pipeline.fit(X, y)
                
                y_pred = pipeline.predict(X)
                accuracy = accuracy_score(y, y_pred)
                f1 = f1_score(y, y_pred, average='weighted')
                
                self.model_scores[name] = {
                    'accuracy': accuracy,
                    'f1': f1,
                    'pipeline': pipeline,
                    'confusion_matrix': confusion_matrix(y, y_pred)
                }
                
                if progress_callback:
                    progress_callback(f"✅ {name} trained successfully! (Accuracy: {accuracy:.4f}, F1: {f1:.4f})")
                
                # Mettre à jour le meilleur modèle
                if f1 > best_f1:
                    best_f1 = f1
                    best_accuracy = accuracy
                    self.best_pipeline = pipeline
                    self.best_model_name = name
                    self.confusion_matrix = confusion_matrix(y, y_pred)
                    
            except Exception as e:
                if progress_callback:
                    progress_callback(f"❌ Error training {name}: {str(e)}")
                continue
        
        self.accuracy = best_accuracy
        self.f1 = best_f1
        
        # Sauvegarder le meilleur pipeline si demandé
        if serialize:
            self._save_pipeline(model_name)
    
    def _save_pipeline(self, model_name: str):
        """Save the trained pipeline and label encoder."""
        # Créer le dossier models s'il n'existe pas
        self.model_path.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le pipeline et l'encodeur
        pipeline_path = self.model_path / f"{model_name}.joblib"
        encoder_path = self.model_path / f"{model_name}_encoder.joblib"
        
        joblib.dump(self.best_pipeline, pipeline_path)
        joblib.dump(self.label_encoder, encoder_path)
    
    def render_confusion_matrix(self):
        """Generate and save confusion matrix plot."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.confusion_matrix,
            annot=True,
            fmt='d',
            cmap='Blues'
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(CM_PLOT_PATH)
        plt.close()
    
    def get_all_scores(self) -> Dict[str, Dict[str, float]]:
        """Return scores for all models."""
        return {
            name: {
                'accuracy': scores['accuracy'],
                'f1': scores['f1']
            }
            for name, scores in self.model_scores.items()
        }

    def get_best_model_name(self) -> str:
        """Return the name of the best performing model."""
        return self.best_model_name


if __name__ == "__main__":
    tp = TrainingPipeline()
    tp.train(serialize=True)
    accuracy, f1_score = tp.get_model_perfomance()
    tp.render_confusion_matrix()
    print(f"ACCURACY = {accuracy}, F1 SCORE = {f1_score}")
