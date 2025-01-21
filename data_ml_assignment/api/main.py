import uvicorn


from data_ml_assignment.api.server import server
from data_ml_assignment.models.database import init_db

if __name__ == "__main__":
    init_db()
    serving_app = server()
    uvicorn.run(
        serving_app,
        host="localhost",
        port=9003,
        log_level="info",
    )
