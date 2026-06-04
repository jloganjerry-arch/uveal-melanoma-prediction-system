import joblib
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
import os

from model.preprocess import load_and_preprocess
from utils.km_plot import generate_km_plot
from utils.survival_matrics import calculate_concordance_index


# ==========================
# 1️⃣ Load and preprocess data
# ==========================
# Make path relative to this script so it runs reliably
base_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_dir, "../data/seer_melanoma.csv")
static_dir = os.path.join(base_dir, "../static/outputs")
os.makedirs(static_dir, exist_ok=True)

df, features = load_and_preprocess(data_path)

X = df[features]
y_time = df["time"]
y_event = df["event"]


# ==========================
# 2️⃣ Train/Test Split
# ==========================

X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
    X, y_time, y_event,
    test_size=0.2,
    random_state=42
)


# ==========================
# 3️⃣ Define XGBoost Cox Model
# ==========================

model = XGBRegressor(
    objective="survival:cox",
    n_estimators=800,
    max_depth=6,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)


# ==========================
# 4️⃣ Train Model
# ==========================

model.fit(X_train, y_time_train)


# ==========================
# 5️⃣ Evaluate Model (C-index on TEST set)
# ==========================

risk_scores_test = model.predict(X_test)

c_index = calculate_concordance_index(y_time_test, y_event_test, risk_scores_test)

print("C-index:", c_index)


# ==========================
# 6️⃣ Generate Kaplan–Meier Plot (FULL dataset)
# ==========================

risk_scores_full = model.predict(X)
km_path = os.path.join(static_dir, "km_plot.png")

generate_km_plot(
    y_time.values,
    y_event.values,
    risk_scores_full,
    km_path
)

print(f"Kaplan-Meier plot saved to {km_path}.")


# ==========================
# 7️⃣ Save Trained Model and Features
# ==========================
model_path = os.path.join(base_dir, "model.pkl")
features_path = os.path.join(base_dir, "features.pkl")

joblib.dump(model, model_path)
joblib.dump(features, features_path)
print(f"Model saved to {model_path}")
print(f"Features saved to {features_path}")
