import os
import sys
import numpy as np
import torch
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier

from quantum_data_processing.data_ingestion import (
    load_data, explore_data, encode_labels, normalize_data, 
    filter_data, merge_data, add_patient_id_column
)
from quantum_data_processing.quantum_algorithms import (
    quantum_clustering_with_error_correction, 
    apply_qpca_with_error_correction, 
    reduce_noise_with_qpca
)
from ai_biomarker_discovery.model_training import create_qnn, train_quantum_model


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

multi_omics_data_paths = [
    "data/genomic_and_protein_metastatic_breast_cancer/brca_data_w_subtypes.csv", 
    "data/metabolic_metastatic_breast_cancer/Coimbra_breast_cancer_dataset.csv"
]


dataframes = load_data(multi_omics_data_paths)
explore_data(dataframes)
dataframes = encode_labels(dataframes)
dataframes = normalize_data(dataframes, method="standard")
dataframes = filter_data(dataframes)
dataframes = reduce_noise_with_qpca(dataframes, n_components=4)
final_data = merge_data(dataframes)

final_data["QuantumCluster"] = quantum_clustering_with_error_correction(final_data.values, n_clusters=2)

processed_data_path = "data/processed_omics_data.csv"
final_data.to_csv(processed_data_path, index=False)
add_patient_id_column(processed_data_path, processed_data_path)
log.info("Preprocessing complete. Quantum PCA and Quantum Clustering applied with error correction. Data saved as processed_omics_data.csv")


MODEL_PATH = "models/NeuralNetworkClassifier_treatment_model_qnn.model"

class TreatmentRecommendation:
    def __init__(self, integrated_data, label_encoder):
        self.data = integrated_data
        self.label_encoder = label_encoder
        self.model = None
        self.qnn = create_qnn(num_qubits=8)
        self.ensure_model_exists()

    def ensure_model_exists(self):
        """Ensure the Quantum Neural Network model is trained and available."""
        if not os.path.exists(MODEL_PATH):
            log.info("No pre-trained model found. Training a new one...")
            self.train_model()
        else:
            log.info("Pre-trained model found. Loading...")
            self.model = NeuralNetworkClassifier.load(MODEL_PATH)

    def train_model(self):
        """Train the Quantum Neural Network (QNN) model."""
        num_qubits = 8

        X = self.data.drop(columns=["PatientId", "Biomarker", "Recommended_Treatment"], errors='ignore')
        y = self.data["Recommended_Treatment"]
        y_encoded = self.label_encoder.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        log.info("Training data sample:\n%s", X_train.head())

        X_train_res = X_train.iloc[:, :num_qubits].astype(np.float32).to_numpy()
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        
        if X_train_res.ndim == 1:
            X_train_res = X_train_res.reshape(-1, num_qubits)

        log.info("Training Quantum Neural Network model...")
        self.model = train_quantum_model(X_train_res, y_train_tensor, data_path=processed_data_path, qnn=self.qnn)
        self.model.save(MODEL_PATH)
        log.info("Model saved to %s", MODEL_PATH)

        X_test_res = X_test.iloc[:, :num_qubits].astype(np.float32).to_numpy()
        if X_test_res.ndim == 1:
            X_test_res = X_test_res.reshape(-1, num_qubits)

        y_pred = self.model.predict(X_test_res)
        accuracy = accuracy_score(y_test, y_pred)
        log.info("Quantum Neural Network Model Accuracy: %.2f%%", accuracy * 100)

    def recommend_treatment(self, patient_features):
        """Predict the best treatment using the trained Quantum Neural Network model."""
        if self.model is None:
            log.info("Loading QNN model for inference...")
            self.model = NeuralNetworkClassifier.load(MODEL_PATH)

        patient_features = np.array(patient_features).reshape(1, -1).astype(np.float32)
        if patient_features.ndim == 1:
            patient_features = patient_features.reshape(-1, len(patient_features))

        predicted_label = self.model.predict(patient_features)[0]
        predicted_label = int(predicted_label)

        if predicted_label not in range(len(self.label_encoder.classes_)):
            raise ValueError(f"Predicted label {predicted_label} is out of range for the encoder.")

        treatment = self.label_encoder.inverse_transform([predicted_label])[0]
        log.info("Recommended Treatment: %s", treatment)
        return treatment
