import streamlit as st
from PIL import Image
import pandas as pd

from data_ml_assignment.dashboard.base import DashboardComponent
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH

class TrainingComponent(DashboardComponent):
    def render(self):
        st.header("Pipeline Training")
        st.info(
            "This section will train multiple models and select the best one. "
            "The models include SVM, Random Forest, Naive Bayes, and others."
        )

        name = st.text_input("Best model save name", placeholder="best_model")
        serialize = st.checkbox("Save best model")
        train = st.button("Train models")

        # Créer un conteneur pour les messages de progression
        progress_container = st.empty()
        
        # Créer un conteneur pour les résultats
        results_container = st.container()

        if train:
            self._handle_training(name, serialize, progress_container, results_container)

    def _handle_training(self, name: str, serialize: bool, 
                        progress_container: st.empty,
                        results_container: st.container):
        """Handle the training process with progress updates."""
        
        # Créer une liste pour stocker les messages de progression
        progress_messages = []
        
        def update_progress(message: str):
            """Callback pour mettre à jour la progression."""
            progress_messages.append(message)
            # Afficher tous les messages de progression
            progress_container.markdown('\n'.join(progress_messages))
        
        with st.spinner("Training multiple models, please wait..."):
            try:
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name, 
                        progress_callback=update_progress)
                
                with results_container:
                    # Afficher les scores de tous les modèles
                    st.subheader("Model Comparison")
                    scores = tp.get_all_scores()
                    df_scores = pd.DataFrame.from_dict(scores, orient='index')
                    st.dataframe(df_scores.style.highlight_max(axis=0))
                    
                    # Afficher le meilleur modèle
                    st.subheader(f"Best Model: {tp.get_best_model_name()}")
                    accuracy, f1 = tp.get_model_perfomance()
                    self._display_metrics(accuracy, f1)
                    
                    # Afficher la matrice de confusion du meilleur modèle
                    tp.render_confusion_matrix()
                    st.image(Image.open(CM_PLOT_PATH), width=850)
                
            except Exception as e:
                self.show_error("Failed to train the models!", e)

    def _display_metrics(self, accuracy: float, f1: float):
        col1, col2 = st.columns(2)
        col1.metric(label="Best Accuracy Score", value=str(round(accuracy, 4)))
        col2.metric(label="Best F1 Score", value=str(round(f1, 4))) 