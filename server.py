"""Local server for the saved heart-disease KNN model."""

from pathlib import Path
import pickle

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent

with open(BASE / "columns.pkl", "rb") as handle:
    FEATURE_COLUMNS = pickle.load(handle)

scaler = joblib.load(BASE / "scaler.pkl")
model = joblib.load(BASE / "KNN_heart.pkl")

CHEST_PAIN = {"ASY", "ATA", "NAP", "TA"}
RESTING_ECG = {"LVH", "Normal", "ST"}
ST_SLOPE = {"Down", "Flat", "Up"}

app = Flask(__name__)


def encode_patient(payload):
    sex = payload["Sex"]
    chest = payload["ChestPainType"]
    ecg = payload["RestingECG"]
    angina = payload["ExerciseAngina"]
    slope = payload["ST_Slope"]

    if sex not in {"M", "F"}:
        raise ValueError("Sex must be M or F.")
    if chest not in CHEST_PAIN:
        raise ValueError("Chest pain type is not recognized.")
    if ecg not in RESTING_ECG:
        raise ValueError("Resting ECG is not recognized.")
    if angina not in {"Y", "N"}:
        raise ValueError("Exercise angina must be Y or N.")
    if slope not in ST_SLOPE:
        raise ValueError("ST slope is not recognized.")

    fasting = int(payload["FastingBS"])
    if fasting not in {0, 1}:
        raise ValueError("Fasting blood sugar must be 0 or 1.")

    values = {
        "Age": float(payload["Age"]),
        "RestingBP": float(payload["RestingBP"]),
        "Cholesterol": float(payload["Cholesterol"]),
        "FastingBS": fasting,
        "MaxHR": float(payload["MaxHR"]),
        "Oldpeak": float(payload["Oldpeak"]),
        "Sex_M": 1 if sex == "M" else 0,
        "ChestPainType_ATA": 1 if chest == "ATA" else 0,
        "ChestPainType_NAP": 1 if chest == "NAP" else 0,
        "ChestPainType_TA": 1 if chest == "TA" else 0,
        "RestingECG_Normal": 1 if ecg == "Normal" else 0,
        "RestingECG_ST": 1 if ecg == "ST" else 0,
        "ExerciseAngina_Y": 1 if angina == "Y" else 0,
        "ST_Slope_Flat": 1 if slope == "Flat" else 0,
        "ST_Slope_Up": 1 if slope == "Up" else 0,
    }

    checks = {
        "Age": (1, 120),
        "RestingBP": (0, 300),
        "Cholesterol": (0, 1000),
        "MaxHR": (30, 250),
        "Oldpeak": (-5, 10),
    }
    for name, (low, high) in checks.items():
        if not low <= values[name] <= high:
            raise ValueError(f"{name} must be between {low} and {high}.")

    frame = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    return frame


@app.get("/")
def index():
    return send_from_directory(BASE, "index.html")


@app.get("/style.css")
def styles():
    return send_from_directory(BASE, "style.css")


@app.get("/app.js")
def scripts():
    return send_from_directory(BASE, "app.js")


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Send a JSON patient record."}), 400

    try:
        frame = encode_patient(payload)
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": str(exc) or "Check the form values."}), 400

    scaled = scaler.transform(frame)
    prediction = int(model.predict(scaled)[0])
    probabilities = model.predict_proba(scaled)[0]
    disease_index = list(model.classes_).index(1)
    probability = float(probabilities[disease_index])
    neighbors_positive = int(round(probability * model.n_neighbors))

    features = {}
    for column in FEATURE_COLUMNS:
        number = frame.iloc[0][column]
        features[column] = int(number) if float(number).is_integer() else round(float(number), 2)

    return jsonify(
        {
            "prediction": prediction,
            "probability": round(probability, 4),
            "neighbors": int(model.n_neighbors),
            "neighbors_positive": neighbors_positive,
            "features": features,
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
