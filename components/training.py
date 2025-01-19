import streamlit as st
from PIL import Image
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH

def show_training():
    st.header("Pipeline Training")
    # Description of the pipeline
    st.subheader("Pipeline Description")
    st.write("""
       This pipeline performs the following steps:
       1. **Data Loading**: The dataset is loaded from the specified path (`RAW_DATASET_PATH`).
       2. **Text Vectorization**: The resume text is converted into numerical features using **TF-IDF Vectorization**.
          - `max_features=5000`: Limits the vocabulary to the top 5,000 terms.
          - `ngram_range=(1, 2)`: Includes both unigrams and bigrams.
       3. **Train-Test Split**: The data is split into training (80%) and testing (20%) sets.
       4. **Model Training**: A **Logistic Regression** model is trained with hyperparameter tuning using **GridSearchCV**.
          - Hyperparameters tuned:
            - `C`: Regularization strength (`[0.1, 1, 10]`).
            - `penalty`: Regularization type (`l1` or `l2`).
          - The model is evaluated using **5-fold cross-validation**.
       5. **Performance Evaluation**: The model's performance is evaluated on both the training and test sets using:
          - **Accuracy**: Proportion of correctly classified resumes.
          - **F1 Score (weighted)**: Balances precision and recall, accounting for class imbalance.
       6. **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance across all classes.
       7. **Model Serialization**: If the `Save pipeline` option is selected, the trained model is saved to a file for future use.
       """)

    # Input fields for pipeline name and save option
    name = st.text_input("Pipeline name", placeholder="logistic_regression")
    serialize = st.checkbox("Save pipeline")
    train = st.button("Train pipeline")

    # Train the pipeline when the button is clicked
    if train:
        with st.spinner("Training pipeline, please wait..."):
            try:
                # Initialize and train the pipeline
                tp = TrainingPipeline()
                tp.train(serialize=serialize, model_name=name)

                # Get training and test performance
                (train_accuracy, train_f1), (test_accuracy, test_f1) = tp.get_model_perfomance()

                # Display training performance metrics
                st.subheader("Training Performance")
                col1, col2 = st.columns(2)
                col1.metric(label="Training Accuracy", value=str(round(train_accuracy, 4)))
                col2.metric(label="Training F1 Score", value=str(round(train_f1, 4)))

                # Display test performance metrics
                st.subheader("Test Performance")
                col3, col4 = st.columns(2)
                col3.metric(label="Test Accuracy", value=str(round(test_accuracy, 4)))
                col4.metric(label="Test F1 Score", value=str(round(test_f1, 4)))

                # Render the confusion matrix
                tp.render_confusion_matrix()

                # Display the confusion matrix image
                st.image(Image.open(CM_PLOT_PATH), width=850)
            except Exception as e:
                st.error("Failed to train the pipeline!")
                st.exception(e)