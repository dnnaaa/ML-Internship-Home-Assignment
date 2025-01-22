from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from data_ml_assignment.models.base_model import BaseModel


class tfidfXGBoost(BaseModel):
    def __init__(self, **kwargs):
        # TfidfVectorizer generates word embeddings for text
        super().__init__(
            model=Pipeline(
                [
                    ("tfidf", TfidfVectorizer(max_features=5000, stop_words='english', **kwargs)),
                    ("classifier", XGBClassifier()),
                ]
            )
        )
