# za korisnicko iskustvo, da bi covek mogao da koristi model

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "deploy.joblib"
ENCODER_PATH = BASE_DIR / "models" / "encoder.joblib"

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

st.title("Flight price prediction App")
st.write("Application for predicting flight prices")

airline = st.selectbox(
    label="Airline",
    options=["AirAsia", "Air_India", "GO_FIRST", "Indigo", "SpiceJet", "Vistara"],
)
source_city = st.selectbox(
    "Source city: ",
    options=["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
)
departure_time = st.selectbox(
    "Departure time",
    options=["Early_Morning", "Morning", "Evening", "Afternoon", "Night", "Late_Night"],
)
stops = st.selectbox("Number of stops", options=["one", "two_or_more", "zero"])
arrival_time = st.selectbox(
    "Arrival time",
    options=["Early_Morning", "Morning", "Evening", "Afternoon", "Night", "Late_Night"],
)
destination_city = st.selectbox(
    "Destination city",
    options=["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"],
)
flight_class = st.selectbox("Flight class", options=["Business", "Economy"])
duration = st.number_input(
    "Duration of flight (hours)",
    min_value=0.0,
    max_value=50.0,
    value=2.0,  # difoltna vrednost
    step=0.5,
    format="%.1f",
)
days_left = st.number_input(
    "Days until flight",
    min_value=1,
    max_value=365,
    value=2,
    step=1,
)

if st.button("Predict"):
    input_data = pd.DataFrame(
        [
            {
                "airline": airline,  #  zbog klase
                "source_city": source_city,
                "departure_time": departure_time,
                "stops": stops,
                "arrival_time": arrival_time,
                "destination_city": destination_city,
                "class": flight_class,
                "duration": duration,
                "days_left": days_left,
            }
        ]
    )

    input_encoded = encoder.transform(input_data)
    prediction = model.predict(input_encoded)[0]  # samo da bismo pristupili prvoj vr

    st.subheader("Results")
    st.write("Prediction: ", prediction)

# POKRETANJE:
# streamlit run app/ui.py
