from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            model=Pipeline([
                ("vectorizer", TfidfVectorizer(max_features=5000, stop_words='english')),
                ("classifier", XGBClassifier(**kwargs))
            ])
        )