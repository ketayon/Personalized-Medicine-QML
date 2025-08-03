import argparse
import logging
import os
import sys
import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "../"))
from treatment_recommendation.recommendation_engine import TreatmentRecommendation
from treatment_recommendation.patient_data_integration import PatientDataIntegration
from workflow.workflow_manager import WorkflowManager

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")

PATIENT_DATA_PATH = os.path.join(DATA_DIR, "patient_profiles.csv")
BIOMARKER_DATA_PATH = os.path.join(DATA_DIR, "identified_quantum_biomarkers.csv")


def recommend_treatment():
    """CLI to get treatment recommendations."""
    data_integrator = PatientDataIntegration(PATIENT_DATA_PATH, BIOMARKER_DATA_PATH)
    integrated_data = data_integrator.integrate_patient_data()
    workflow_manager = WorkflowManager(integrated_data, data_integrator.label_encoder)

    log.info("Enter patient biomarker data (comma-separated):")
    patient_input = input("> ").strip()
    patient_features = np.array([float(x) for x in patient_input.split(",")]).reshape(1, -1)

    recommended_treatment = workflow_manager.infer_treatment(patient_features)
    log.info(f"Recommended Treatment: {recommended_treatment}")


def main():
    parser = argparse.ArgumentParser(description="CLI for Quantum AI-Based Drug Discovery")
    parser.add_argument("--recommend", action="store_true", help="Get a treatment recommendation")

    args = parser.parse_args()

    if args.recommend:
        recommend_treatment()


if __name__ == "__main__":
    main()
