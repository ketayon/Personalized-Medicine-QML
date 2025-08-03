import pytest
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from ai_biomarker_discovery.preprocessing import load_data, preprocess_data
from ai_biomarker_discovery.model_training import create_qnn, train_quantum_model
from ai_biomarker_discovery.evaluation import evaluate_model, identify_biomarkers


### **TEST 1: Data Loading**
def test_load_data():
    data_path = "./data/processed_omics_data.csv"
    X, y = load_data(data_path)
    
    assert isinstance(X, pd.DataFrame), "X should be a DataFrame"
    assert isinstance(y, pd.Series), "y should be a Series"
    assert not X.empty, "X should not be empty"
    assert not y.empty, "y should not be empty"


### **TEST 2: Data Preprocessing**
def test_preprocess_data():
    data_path = "./data/processed_omics_data.csv"
    X, y = load_data(data_path)
    X_train, X_test, y_train_tensor, y_test_tensor = preprocess_data(X, y)

    assert isinstance(X_train, pd.DataFrame), "X_train should be a DataFrame"
    assert isinstance(X_test, pd.DataFrame), "X_test should be a DataFrame"
    assert isinstance(y_train_tensor, torch.Tensor), "y_train_tensor should be a torch Tensor"
    assert isinstance(y_test_tensor, torch.Tensor), "y_test_tensor should be a torch Tensor"
    assert X_train.shape[0] > 0, "X_train should have samples"
    assert X_test.shape[0] > 0, "X_test should have samples"


### **TEST 3: Quantum Model Creation**
def test_create_qnn():
    qnn = create_qnn()
    assert qnn is not None, "QNN should be created successfully"


### **TEST 4: Model Training**
@pytest.fixture
def trained_model():
    data_path = "../data/processed_omics_data.csv"
    data = pd.read_csv(data_path)
    X = data.drop(columns=["QuantumCluster"])
    y = data["QuantumCluster"]
    num_qubits = 8
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_res = X_train.iloc[:, :num_qubits].to_numpy()
    X_test_res = X_test.iloc[:, :num_qubits].to_numpy()
    y_test = y_test.iloc[8]
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    classifier = NeuralNetworkClassifier.load('./models/classifier.model')
    
    assert isinstance(classifier, NeuralNetworkClassifier), "Classifier should be a NeuralNetworkClassifier"
    return classifier, X_test, y_test_tensor


### **TEST 5: Model Evaluation**
def test_evaluate_model():
    data_path = "./data/processed_omics_data.csv"
    
    # Ensure file exists
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.fail(f"Missing test data file: {data_path}")

    X = data.drop(columns=["QuantumCluster"])
    y = data["QuantumCluster"]
    num_qubits = 8  # Ensure we use only 8 features

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure shape matches model input
    X_test_res = X_test.iloc[:, :num_qubits].to_numpy()
    y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.float32)  # Keep all test labels

    # Load trained classifier
    try:
        classifier = NeuralNetworkClassifier.load('./models/classifier.model')
    except FileNotFoundError:
        pytest.fail("Model file missing: Ensure './models/classifier.model' exists")

    try:
        evaluate_model(classifier, X_test_res, y_test_tensor)
    except Exception as e:
        pytest.fail(f"Model evaluation failed: {e}")



### **TEST 6: Biomarker Identification**
def test_identify_biomarkers():
    data_path = "./data/processed_omics_data.csv"

    # Ensure file exists
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        pytest.fail(f"Missing test data file: {data_path}")

    X = data.drop(columns=["QuantumCluster"])
    y = data["QuantumCluster"]
    num_qubits = 8  # Ensure correct feature size

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    try:
        quantum_feature_importance = np.abs(X_train.iloc[:, :num_qubits].mean(axis=0) - X_test.iloc[:, :num_qubits].mean(axis=0))
        feature_names = X.columns[:num_qubits]
        feature_importance = pd.Series(quantum_feature_importance, index=feature_names).sort_values(ascending=False)

        print("Top Quantum Biomarkers Identified:")
        print(feature_importance.head(10))

        # Save test biomarker file
        biomarker_path = "./data/test_identified_quantum_biomarkers.csv"
        pd.DataFrame({
            "Biomarker": feature_importance.head(10).index.tolist(),
            "Value": feature_importance.head(10).values
        }).to_csv(biomarker_path, index=False)

        print(f"Biomarkers saved in {biomarker_path}")
    except Exception as e:
        pytest.fail(f"Biomarker identification failed: {e}")

