# backend/app/model.py

import joblib
from pathlib import Path

# Get absolute path to the model file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent / "models" / "spam_classifier.pkl"

# Load the model
model = joblib.load(MODEL_PATH)

def predict_spam(message: str) -> str:
    pred = model.predict([message])[0]
    return "spam" if pred == 1 else "ham"
