from fastapi import FastAPI

from data_ml_assignment.api.inference_route import inference_router
from data_ml_assignment.api.constants import APP_NAME, API_PREFIX


def server() -> FastAPI:
    app = FastAPI(
        title=APP_NAME,
        docs_url=f"{API_PREFIX}/docs",
    )
    app.include_router(inference_router, prefix=API_PREFIX)
    
    # Ajouter une route pour `/`
    @app.get("/")
    async def root():
        return {"message": "Welcome to the API!"}

    
    return app
