import logging
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def evaluate_model(classifier, X_test, y_test):
    """Evaluate the Quantum Neural Network model."""
    y_pred = classifier.predict(X_test)
    accuracy = np.mean(y_pred == y_test.numpy())
    
    log.info("Final Quantum Model Accuracy: %.2f%%", accuracy * 100)
    log.info("Quantum Neural Network Performance:\n%s", classification_report(y_test.numpy(), y_pred))


def identify_biomarkers(X_train, X_test, data_path, feature_names):
    """Identify key quantum biomarkers using feature importance analysis."""
    num_qubits = 8
    data = pd.read_csv(data_path)
    X = data.drop(columns=["QuantumCluster"])
    
    quantum_feature_importance = np.abs(X_train.mean(axis=0) - X_test.mean(axis=0))
    feature_names = X.columns[:num_qubits]
    
    feature_importance = pd.Series(quantum_feature_importance, index=feature_names).sort_values(ascending=False)
    
    log.info("Top Quantum Biomarkers Identified:\n%s", feature_importance.head(10))

    top_biomarkers = feature_importance.head(10).index.tolist()
    biomarker_df = pd.DataFrame({
        "Biomarker": top_biomarkers,
        "Value": feature_importance.head(10).values
    })
    
    biomarker_df.to_csv("../data/identified_quantum_biomarkers.csv", index=False)
    
    log.info("Quantum Biomarker Discovery Complete. Identified biomarkers saved to '../data/identified_quantum_biomarkers.csv'.")
