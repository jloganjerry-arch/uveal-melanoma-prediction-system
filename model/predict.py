import joblib
import numpy as np
import os
from model.preprocess import load_and_preprocess

base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
features_path = os.path.join(base_dir, "features.pkl")

# Load trained model once
model = joblib.load(model_path)
try:
    features = joblib.load(features_path)
except:
    features = ['Sex', 'Age', 'Race', 'Stage', 'Radiation', 'Chemotherapy']

def predict_risk(input_data):
    """
    input_data: list in same order as features
    Returns predicted risk score, the model itself, and feature names.
    """

    input_array = np.array([input_data])
    risk_score = model.predict(input_array)

    return float(risk_score[0]), model, features
