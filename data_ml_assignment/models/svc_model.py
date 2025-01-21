from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
import re
from data_ml_assignment.models.base_model import BaseModel


# class SVCModel(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(model=Pipeline([("svc", SVC(**kwargs))]))

class SVCModel(BaseModel):
    def __init__(self, **kwargs):
        def preprocess_text(text):
            if not isinstance(text, str):
                return ""
            
            text = text.lower()
            text = re.sub(r'[^\w\s]', ' ', text)
            text = ' '.join(text.split())
            return text
            
        super().__init__(
            model=Pipeline([
                ('tfidf', TfidfVectorizer(
                    max_features=5000, 
                    min_df=5,
                    max_df=0.90,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm='l2',
                    stop_words='english'
                )),
                ('svc', SVC(
                    C=1.0,
                    kernel='linear',
                    class_weight='balanced',
                    probability=True,
                    max_iter=1000,
                    tol=1e-4,
                    random_state=42,
                    **kwargs
                ))
            ])
        )