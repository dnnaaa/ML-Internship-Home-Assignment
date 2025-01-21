from pydantic import BaseModel

class Resume(BaseModel):
    text: str
    
# Response model for predictions
class PredictionResponse(BaseModel):
    text: str
    predict: str
