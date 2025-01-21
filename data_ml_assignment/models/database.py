from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from pathlib import Path

from data_ml_assignment.constants import DATABASE_URL

# Make sure the database directory exists
db_path = Path(DATABASE_URL.replace('sqlite:///', ''))
db_path.parent.mkdir(parents=True, exist_ok=True)

Base = declarative_base()

class PredictionResult(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    resume_text = Column(String, nullable=False)
    prediction = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create engine with correct URL
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})

# Create all tables
Base.metadata.create_all(bind=engine)
print("Database initialized successfully.")

# SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)