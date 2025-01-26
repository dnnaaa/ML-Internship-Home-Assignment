import streamlit as st
from data_ml_assignment.training.train_pipeline import NaiveBayesPipeline, SVMPipeline

def training_section():
    st.title("Model Training")

    # Sélection du modèle
    model_option = st.selectbox("Choose Model", ("Naive Bayes", "SVM"))

    # Demander à l'utilisateur s'il souhaite sauvegarder le modèle avant l'entraînement
    save_model = st.checkbox("Save the model after training?")
    
    model_name = ""
    if save_model:
        # Demander un nom pour le modèle si l'utilisateur souhaite le sauvegarder
        model_name = st.text_input("Enter model name:", "trained_model")

    # Chargement des données et préparation du pipeline en fonction du modèle sélectionné
    if model_option == "Naive Bayes":
        model_pipeline = NaiveBayesPipeline()
    elif model_option == "SVM":
        model_pipeline = SVMPipeline()  # Assurez-vous d'avoir la classe SVM définie dans votre projet

    # Bouton pour entraîner le modèle
    if st.button("Train Model"):
        # Charger les données (chemin relatif vers votre dataset)
        data_path = "data/raw/resume.csv"  # Utilisez un chemin relatif ici
        data = model_pipeline.load_data(data_path)

        if data is not None:
            # Entraîner le modèle et obtenir les résultats
            accuracy, f1 = model_pipeline.train(serialize=True)

            # Affichage des résultats dans Streamlit
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"F1 Score: {f1:.2f}")
            
            # Afficher la matrice de confusion
            model_pipeline.render_confusion_matrix()  # Affiche la matrice de confusion après l'entraînement

            # Si l'utilisateur a choisi de sauvegarder le modèle, procéder à la sauvegarde
            if save_model:
                if model_name:
                    # Sauvegarder le modèle avec le nom spécifié
                    model_pipeline.serialize_model(model_name)
                    st.success(f"Model saved as {model_name}.pkl")
                else:
                    st.error("Please provide a model name before saving.")
            
        else:
            st.error("Error loading data. Please check the file path and format.")
