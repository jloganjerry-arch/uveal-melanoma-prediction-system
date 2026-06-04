import os
import json
from flask import Flask, render_template, request, jsonify
from model.predict import predict_risk
from database.create_db import db, init_db, Prediction
from utils.shap_analysis import generate_shap_plot
from config import Config

app = Flask(__name__)
app.config.from_object(Config)

# Initialize DB
init_db(app)

# Ensure static outputs folder exists
STATIC_OUTPUTS = os.path.join(app.root_path, 'static', 'outputs')
os.makedirs(STATIC_OUTPUTS, exist_ok=True)

# ==============================
# Single Page Application View
# ==============================
@app.route("/")
@app.route("/login")
def index():
    return render_template("login.html")

# ==============================
# API: Make Prediction
# ==============================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    km_plot_url = "outputs/km_plot.png" if os.path.exists(os.path.join(STATIC_OUTPUTS, 'km_plot.png')) else None

    try:
        data = request.get_json()
        sex = int(data.get("sex", 0))
        age = int(data.get("age", 0))
        race = int(data.get("race", 0))
        stage = int(data.get("stage", 0))
        radiation = int(data.get("radiation", 0))
        chemo = int(data.get("chemo", 0))

        input_data = [sex, age, race, stage, radiation, chemo]

        # Get Prediction, model, and feature names
        risk_score_raw, model, features = predict_risk(input_data)
        risk_score = float(risk_score_raw)
        
        prediction_val = round(risk_score, 2)
        
        if risk_score > 70:
            risk_level = "High Risk"
        elif risk_score >= 50:
            risk_level = "Intermediate Risk"
        else:
            risk_level = "Low Risk"

        # Generate SHAP Plot for this patient
        shap_filename = "shap_plot.png"
        shap_path = os.path.join(STATIC_OUTPUTS, shap_filename)
        generate_shap_plot(model, input_data, features, shap_path)
        shap_url = f"outputs/{shap_filename}"
        
        # Save to Database
        new_pred = Prediction(
            sex=sex, age=age, race=race, stage=stage, 
            radiation=radiation, chemo=chemo, 
            risk_score=prediction_val, risk_category=risk_level
        )
        db.session.add(new_pred)
        db.session.commit()

        return jsonify({
            "success": True,
            "prediction": prediction_val,
            "risk_level": risk_level,
            "shap_url": shap_url,
            "km_plot_url": km_plot_url
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ==============================
# API: Get Records Data
# ==============================
@app.route("/api/records", methods=["GET"])
def api_records():
    records = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    
    km_plot_url = "outputs/km_plot.png" if os.path.exists(os.path.join(STATIC_OUTPUTS, 'km_plot.png')) else None

    # Serialize records to simple dict array
    record_list = [{
        "id": r.id,
        "timestamp": r.timestamp.strftime('%Y-%m-%d %H:%M'),
        "sex": r.sex,
        "age": r.age,
        "race": r.race,
        "stage": r.stage,
        "radiation": r.radiation,
        "chemo": r.chemo,
        "risk_score": r.risk_score,
        "risk_category": r.risk_category
    } for r in records]

    return jsonify({
        "success": True,
        "records": record_list,
        "km_plot_url": km_plot_url
    })

# ==============================
# API: Get Patients Data from CSV
# ==============================
@app.route("/api/patients", methods=["GET"])
def api_patients():
    import csv
    patients = []
    csv_path = os.path.join(app.root_path, 'data', 'seer_melanoma.csv')
    try:
        with open(csv_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            count = 0
            for row in reader:
                
                
                patient = {
                    "id": f"{count+1}A",  # Generating a mock ID like 1A, 2A to match the ref image
                    "sex": row.get("Sex", "Unknown"),
                    "age": row.get("Age recode with <1 year olds and 90+", "Unknown"),
                    "race": row.get("Race recode (W, B, AI, API)", "Unknown"),
                    "year": row.get("Year of diagnosis", "Unknown"),
                    "stage": row.get("Derived AJCC Stage Group, 6th ed (2004-2015)", "Unknown"),
                    "survival": row.get("Survival months", "Unknown"),
                    "status": row.get("Vital status recode (study cutoff used)", "Unknown"),
                    "primary_site": row.get("Primary Site - labeled", "Unknown"),
                    "histologic_type": row.get("Histologic Type ICD-O-3", "Unknown"),
                    "marital_status": row.get("Marital status at diagnosis", "Unknown")
                }
                patients.append(patient)
                count += 1
                
        return jsonify({
            "success": True,
            "patients": patients
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

# ==============================
# Run App
# ==============================
if __name__ == "__main__":
    app.run(debug=True)

