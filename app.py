
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import pandas as pd
import joblib

app = FastAPI(title="Milk Batch Rejection Prediction API")

model = joblib.load("milk_rejection_model.pkl")


class MilkBatch(BaseModel):

    milk_quantity: float = Field(gt=0, le=100000)
    collection_temp: float = Field(ge=0, le=60)
    transport_duration: float = Field(ge=0, le=48)
    cooling_delay: float = Field(ge=0, le=24)
    fat_percentage: float = Field(ge=0, le=15)
    acidity_indicator: float = Field(ge=0, le=14)
    container_type: Literal["Steel", "Plastic", "Aluminium"]
    ambient_temp: float = Field(ge=-10, le=60)


def create_features(data):

    data["temperature_exposure_index"] = (
        (data["collection_temp"] + data["ambient_temp"]) / 2
    ) * (1 + data["cooling_delay"] / 5)

    data["transport_delay_risk"] = (
        data["transport_duration"] / 2
        + data["cooling_delay"]
    )

    return data


@app.get("/")
def home():

    return {
        "message": "Milk Batch Rejection Prediction API is running"
    }


@app.post("/predict")
def predict(batch: MilkBatch):

    try:

        data = pd.DataFrame([batch.model_dump()])

        data = create_features(data)

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Rejection Risk"
        else:
            result = "Likely Accepted"

        return {
            "prediction": result
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
