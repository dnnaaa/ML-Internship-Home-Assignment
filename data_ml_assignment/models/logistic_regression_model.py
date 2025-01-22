from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class LogisticRegressionModel(BaseModel):
    def __init__(self):
        # Create full pipeline first
        pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english')),
            ("lr", LogisticRegression(max_iter=1000))
        ])
        
        # Pass pipeline to BaseModel
        super().__init__(model=pipeline)