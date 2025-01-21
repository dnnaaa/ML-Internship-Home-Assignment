from sklearn.svm import LinearSVC # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.calibration import CalibratedClassifierCV # type: ignore
import re
from data_ml_assignment.models.base_model import BaseModel



class SVMModel(BaseModel):
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
                    max_df=0.95,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm='l2',
                    stop_words='english'
                )),
                
                ('svm', CalibratedClassifierCV(
                    LinearSVC(
                        C=1.0,  
                        class_weight='balanced',
                        dual=False,  
                        max_iter=1000,
                        tol=1e-4,
                        **kwargs
                    ), cv=3))
            ])
        )