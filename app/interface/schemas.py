from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predictions: list[list[float]]
