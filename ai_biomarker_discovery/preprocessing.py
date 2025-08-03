import logging
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def load_data(data_path):
    """Load dataset from a CSV file and separate features from labels."""
    log.info("Loading dataset from %s", data_path)
    
    data = pd.read_csv(data_path)

    if "QuantumCluster" not in data.columns:
        raise KeyError("Missing 'QuantumCluster' column in dataset.")

    X = data.drop(columns=["QuantumCluster"])
    y = data["QuantumCluster"]

    log.info("Dataset loaded successfully. Shape: %s, Target variable: 'QuantumCluster'", X.shape)
    
    return X, y


def preprocess_data(X, y, test_size=0.2, random_state=42):
    """Preprocess dataset by splitting into training and testing sets."""
    log.info("Splitting dataset into train and test sets (test size = %.2f)", test_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    log.info("Training set shape: %s, Testing set shape: %s", X_train.shape, X_test.shape)

    return (
        X_train, 
        X_test, 
        torch.tensor(y_train.to_numpy(), dtype=torch.float32), 
        torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    )
