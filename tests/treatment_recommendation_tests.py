import os
import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import LabelEncoder
from treatment_recommendation.patient_data_integration import PatientDataIntegration
from treatment_recommendation.recommendation_engine import TreatmentRecommendation

TEST_PATIENT_DATA_PATH = "tests/test_patient_profiles.csv"
TEST_BIOMARKER_DATA_PATH = "tests/test_biomarker_data.csv"
MODEL_PATH = "tests/models/NeuralNetworkClassifier_treatment_model_qnn.model"


@pytest.fixture
def mock_patient_data():
    """Creates a mock patient data DataFrame."""
    data = {
        "PatientId": ["0000001", "0000002", "0000003"],
        "Age": [50, 60, 45],
        "Tumor_Stage": ["Stage II", "Stage III", "Stage I"],
        "Hormone_Receptor_Status": ["Positive", "Negative", "Positive"],
        "HER2_Status": ["Negative", "Positive", "Negative"],
        "Genetic_Risk_Score": [0.7, 0.4, 0.9],
        "Recommended_Treatment": ["Chemotherapy", "Targeted Therapy", "Hormonal Therapy"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_biomarker_data():
    """Creates a mock biomarker data DataFrame."""
    data = {
        "Biomarker": ["0000001", "0000002", "0000003"],
        "Value": [0.5, 0.8, 0.3]
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_data_integration(mock_patient_data, mock_biomarker_data):
    """Mocks PatientDataIntegration class with mock patient and biomarker data."""
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.side_effect = [mock_patient_data, mock_biomarker_data]  # Mock patient & biomarker CSVs
        return PatientDataIntegration(TEST_PATIENT_DATA_PATH, TEST_BIOMARKER_DATA_PATH)


@pytest.fixture
def mock_treatment_recommendation(mock_data_integration):
    """Creates a mock TreatmentRecommendation instance with test patient data."""
    integrated_data = mock_data_integration.integrate_patient_data()
    return TreatmentRecommendation(integrated_data, mock_data_integration.label_encoder)


def test_patient_data_integration(mock_data_integration):
    """Test patient data integration process."""
    integrated_data = mock_data_integration.integrate_patient_data()

    assert "PatientId" in integrated_data.columns
    assert "Value" in integrated_data.columns
    assert "Recommended_Treatment" in integrated_data.columns
    assert integrated_data.shape[0] == 3


def test_recommend_treatment(mock_treatment_recommendation):
    """Test treatment recommendation based on patient features."""
    patient_features = [0.6, 0.9, 0.5, 0.7, 1.1, 0.2, 0.4, 0.6]
    
    with patch("os.path.exists", return_value=True), \
         patch("qiskit_machine_learning.algorithms.classifiers.NeuralNetworkClassifier.load", return_value=MagicMock()) as mock_load:
        
        mock_treatment_recommendation.model = mock_load()
        mock_treatment_recommendation.model.predict.return_value = np.array([1])
        mock_treatment_recommendation.label_encoder.inverse_transform = MagicMock(return_value=["Chemotherapy"])
        
        recommended_treatment = mock_treatment_recommendation.recommend_treatment(patient_features)
        
        assert recommended_treatment == "Chemotherapy"
        mock_treatment_recommendation.model.predict.assert_called_once()


if __name__ == "__main__":
    pytest.main()
