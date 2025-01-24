from sqlalchemy import Column, Integer, String, DateTime, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class Prediction(Base):
    """Model for storing prediction results."""
    __tablename__ = 'predictions'

    id = Column(Integer, primary_key=True)
    resume_type = Column(String, nullable=False)
    predicted_label = Column(String, nullable=False)
    prediction_time = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<Prediction(resume_type='{self.resume_type}', predicted_label='{self.predicted_label}')>"

# Création de la base de données et des tables
engine = create_engine('sqlite:///predictions.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine) 