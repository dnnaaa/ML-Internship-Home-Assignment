from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class Prediction(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    resume = Column(String)
    predicted_label = Column(String)
    prediction_time = Column(DateTime, default=datetime.utcnow)

# Ensure the database directory exists
os.makedirs('database', exist_ok=True)

# Create engine and session
engine = create_engine('sqlite:///database/predictions.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

def add_prediction_to_db(resume_text, predicted_label):
    """Add a new prediction to the database"""
    session = Session()
    new_prediction = Prediction(
        resume=resume_text, 
        predicted_label=predicted_label
    )
    session.add(new_prediction)
    session.commit()
    session.close()

def get_all_predictions():
    """Retrieve all predictions from the database"""
    session = Session()
    predictions = session.query(Prediction).all()
    session.close()
    return predictions