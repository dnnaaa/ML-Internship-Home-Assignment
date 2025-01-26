from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Table, MetaData
from sqlalchemy.orm import sessionmaker


# Database setup
DATABASE_URL = "sqlite:///./predictions.db"
engine = create_engine(DATABASE_URL)
metadata = MetaData()


# Define the predictions table
predictions_table = Table(
    "predictions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("input_data", String, nullable=False),
    Column("prediction", String, nullable=False)
)


# Create the table if it doesn't exist
metadata.create_all(engine)

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Helper function to save a prediction
def save_prediction_to_db(input_data: str, prediction: str):
    session = SessionLocal()
    try:
        insert_query = predictions_table.insert().values(
            input_data=input_data,
            prediction=prediction,
            
        )
        session.execute(insert_query)
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


# Helper function to retrieve all predictions
def get_all_predictions():
    session = SessionLocal()
    try:
        select_query = predictions_table.select()
        results = session.execute(select_query).fetchall()
        return [
            {"id": row.id, "input_data": row.input_data, "prediction": row.prediction}
            for row in results
        ]
    except Exception as e:
        raise e
    finally:
        session.close()

def delete_prediction(id):
    session = SessionLocal()
    try:
        delete_query = predictions_table.delete().where(predictions_table.c.id == id)  
        session.execute(delete_query)
        session.commit()
    except Exception as e:
        raise e
    finally:
        session.close()
