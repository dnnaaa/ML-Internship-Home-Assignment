import streamlit as st
import time
from data_ml_assignment.training.train_pipeline import TrainingPipeline
from data_ml_assignment.constants import CM_PLOT_PATH

MODEL_DESCRIPTIONS = {
    "naive_bayes": "Baseline model using Naive Bayes with CountVectorizer",
    "svc": "Support Vector Classifier with TF-IDF features",
    "xgboost": "Gradient Boosted Trees with TF-IDF features",
    "logistic_regression": "Logistic Regression with TF-IDF features"
}

def display_training_section():
    st.header("🤖 Model Training")
    
    # Model selection with descriptions
    model_type = st.selectbox(
        "Select Model Architecture",
        options=list(TrainingPipeline.MODEL_REGISTRY.keys()),
        format_func=lambda x: f"{x.title()} - {MODEL_DESCRIPTIONS[x]}"
    )
    
    with st.form("training_config"):
        model_name = st.text_input(
            "Model Name", 
            value=f"{model_type}_pipeline",
            help="Name for saving the trained model"
        )
        
        save_model = st.checkbox(
            "Save Model", 
            value=True,
            help="Save model to disk for later use"
        )
        
        if st.form_submit_button("🚀 Start Training"):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Initialize pipeline
                status_text.markdown("🔄 Initializing training environment...")
                tp = TrainingPipeline(model_type=model_type)
                progress_bar.progress(10)
                
                # Training phase
                status_text.markdown("🔥 Training model...")
                start_time = time.time()
                tp.train(serialize=save_model, model_name=model_name)
                training_time = time.time() - start_time
                progress_bar.progress(70)
                
                # Evaluation phase
                status_text.markdown("📊 Evaluating performance...")
                accuracy, f1 = tp.get_model_perfomance()
                progress_bar.progress(90)
                
                # Results display
                status_text.markdown("✅ Training complete!")
                progress_bar.progress(100)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Accuracy", f"{accuracy:.2%}")
                col2.metric("F1 Score", f"{f1:.2%}")
                col3.metric("Training Time", f"{training_time:.1f}s")
                
                with st.expander("Confusion Matrix"):
                    tp.render_confusion_matrix()
                    st.image(str(CM_PLOT_PATH), use_column_width=True)
                    
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
                st.exception(e)
                progress_bar.empty()
                status_text.markdown("🛑 Training aborted due to errors")