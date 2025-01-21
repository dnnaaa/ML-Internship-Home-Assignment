import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class XGBCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=Pipeline([('tfidf', TfidfVectorizer()),("xgbc", xgb.XGBClassifier(**kwargs))]))
