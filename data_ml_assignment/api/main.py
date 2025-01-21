import uvicorn

from data_ml_assignment.api.server import server

if __name__ == "__main__":
    serving_app = server()
    uvicorn.run(
        serving_app,
        host="localhost",
        port=9001,
        log_level="info",
    )
