import os
import sys
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

# Ensure the correct module path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "../../"))  # Move up two levels to access modules

from treatment_recommendation.recommendation_engine import TreatmentRecommendation
from treatment_recommendation.patient_data_integration import PatientDataIntegration
from workflow.workflow_manager import WorkflowManager

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../../data")

PATIENT_DATA_PATH = os.path.join(DATA_DIR, "patient_profiles.csv")
BIOMARKER_DATA_PATH = os.path.join(DATA_DIR, "identified_quantum_biomarkers.csv")

data_integrator = PatientDataIntegration(PATIENT_DATA_PATH, BIOMARKER_DATA_PATH)
integrated_data = data_integrator.integrate_patient_data()
workflow_manager = WorkflowManager(integrated_data, data_integrator.label_encoder)

@app.route("/")
def home():
    """Render the main dashboard."""
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend():
    """API Endpoint for Treatment Recommendation."""
    try:
        patient_data = request.json.get("patient_features", [])
        patient_features = np.array(patient_data).reshape(1, -1)
        recommended_treatment = workflow_manager.infer_treatment(patient_features)
        return jsonify({"treatment": recommended_treatment})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/patient-data")
def patient_data():
    """API to fetch patient data for visualization."""
    df = pd.read_csv(PATIENT_DATA_PATH)
    return jsonify(df.to_dict(orient="records"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)

