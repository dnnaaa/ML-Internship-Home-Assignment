from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import inspect

DATABASE_PATH = 'predictions.db'  # Path to your SQLite database
Base = declarative_base()

class Prediction(Base):
    """Model for storing predictions in the database."""
    __tablename__ = 'predictions'
    id = Column(Integer, primary_key=True)
    sample_name = Column(String, nullable=False)
    prediction_label = Column(String, nullable=False)

def create_database():
    """Create database and table if they don't exist."""
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    with engine.connect() as connection:
        inspector = inspect(connection)
        if not inspector.has_table('predictions'):
            Base.metadata.create_all(engine)
            print("Database and table created!")
        else:
            print("Table already exists!")

def save_prediction_to_db(sample_name: str, prediction_label: str):
    """Save prediction result to the database."""
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session: Session = SessionLocal()

    # Add the new prediction to the database
    prediction = Prediction(sample_name=sample_name, prediction_label=prediction_label)
    session.add(prediction)
    session.commit()
    print(f"Prediction for {sample_name} saved to DB.")
    session.close()

def get_predictions():
    """Fetch all predictions from the database."""
    engine = create_engine(f"sqlite:///{DATABASE_PATH}")
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session: Session = SessionLocal()

    predictions = session.query(Prediction).all()
    session.close()
    return predictions
