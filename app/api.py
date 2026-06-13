from pathlib import Path

import pandas as pd
import joblib

from fastapi import FastAPI
from pydantic import BaseModel, Field

# api - da bi druga masina mogla da koristi nas model
# APIjem povezujemo vise mikroservisa

# uvicorn app.api:app --reload je pokretanje IZ ROOTA PROJEKTA, moras uv add uvicorn pre toga
# nalepiti ovo na kraj linka apija za onaj lepsi prikaz: /docs#/default/predict_predict_post

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "deploy.joblib"
ENCODER_PATH = BASE_DIR / "models" / "encoder.joblib"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)


class FlightInput(BaseModel):
    airline: str
    source_city: str
    departure_time: str
    stops: str
    arrival_time: str
    destination_city: str
    flight_class: str = Field(alias="class")
    duration: float
    days_left: int


@app.get("/")
def home():
    return {"message": "Flight price prediction API is running ..."}


@app.post("/predict")
def predict(flight: FlightInput):
    input_Data = pd.DataFrame(
        [
            {
                "airline": flight.airline,  # flight. zbog klase
                "source_city": flight.source_city,
                "departure_time": flight.departure_time,
                "stops": flight.stops,
                "arrival_time": flight.arrival_time,
                "destination_city": flight.destination_city,
                "class": flight.flight_class,
                "duration": flight.duration,
                "days_left": flight.days_left,
            }
        ]
    )

    # Enkodiraj kategoričke kolone pre predikcije
    input_encoded = encoder.transform(input_Data)
    prediction = model.predict(input_encoded)
    return {"prediction": prediction[0]}
