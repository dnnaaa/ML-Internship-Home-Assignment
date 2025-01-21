import uvicorn
from data_ml_assignment.api.server import server

if __name__ == "__main__":
    # Start the FastAPI server
    serving_app = server()
    uvicorn.run(
        serving_app,
        host="localhost",
        port=9000,
        log_level="info",
    )

