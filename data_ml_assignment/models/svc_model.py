from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from data_ml_assignment.models.base_model import BaseModel


class SVCModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(model=Pipeline([('tfidf', TfidfVectorizer()),("svc", SVC(**kwargs))]))
