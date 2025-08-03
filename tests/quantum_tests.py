import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from quantum_data_processing.quantum_algorithms import (
    apply_qpca_with_error_correction, 
    quantum_kernel,
    quantum_clustering_with_error_correction,
    reduce_noise_with_qpca
)
from quantum_data_processing.noise_mitigation import (
    local_folding, 
    linear_extrapolation, 
    quantum_kernel_with_error_correction
)
from quantum_data_processing.data_ingestion import (
    load_data, 
    encode_labels, 
    normalize_data, 
    merge_data
)

def test_apply_qpca_with_error_correction():
    """Test Quantum PCA with simulated input."""
    data = np.random.rand(10, 8)  # 10 samples, 8 features
    reduced_data = apply_qpca_with_error_correction(data, n_components=4)
    
    assert reduced_data.shape == (10, 4), "QPCA output shape mismatch."


def test_quantum_clustering_with_error_correction():
    """Test Quantum Clustering with dummy data."""
    data = np.random.rand(10, 4)  # 10 samples, 4 features
    clusters = quantum_clustering_with_error_correction(data, n_clusters=2)
    
    assert len(clusters) == 10, "Cluster labels length mismatch."
    assert set(clusters) == {0, 1}, "Clusters must be 0 or 1."

def test_reduce_noise_with_qpca():
    """Test Quantum PCA noise reduction on synthetic dataset."""
    df = pd.DataFrame(np.random.rand(10, 6), columns=[f'feature_{i}' for i in range(6)])
    dataframes = {"test": df}
    reduced_dataframes = reduce_noise_with_qpca(dataframes, n_components=3)
    
    assert reduced_dataframes["test"].shape == (10, 3), "QPCA reduced dataset has incorrect dimensions."

def test_local_folding():
    """Test noise mitigation local folding (mock)."""
    assert callable(local_folding), "local_folding must be callable."

def test_linear_extrapolation():
    """Test linear extrapolation noise mitigation."""
    scale_factors = [1, 2, 3]
    results = [0.1, 0.2, 0.3]
    estimated_zero_noise = linear_extrapolation(scale_factors, results)
    
    assert isinstance(estimated_zero_noise, float), "Linear extrapolation must return a float."

def test_quantum_kernel_with_error_correction():
    """Test quantum kernel with error correction."""
    x1 = np.random.rand(4)
    x2 = np.random.rand(4)
    
    similarity = quantum_kernel_with_error_correction(x1, x2)
    assert isinstance(similarity, float), "Quantum kernel with noise correction should return a float."
    assert 0 <= similarity <= 1, "Quantum kernel similarity should be between 0 and 1."

def test_load_data():
    """Test loading of synthetic CSV data."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    df.to_csv("test_data.csv", index=False)
    
    loaded_data = load_data(["test_data.csv"])
    assert "test_data.csv" in loaded_data, "Data loading failed."
    assert loaded_data["test_data.csv"].shape == (3, 2), "Data shape mismatch."

def test_encode_labels():
    """Test label encoding."""
    df = pd.DataFrame({"Category": ["A", "B", "A", "C"]})
    dataframes = {"test": df}
    encoded_data = encode_labels(dataframes)
    
    assert encoded_data["test"].dtypes["Category"] == np.int64, "Encoding failed."

def test_normalize_data():
    """Test data normalization."""
    df = pd.DataFrame(np.random.rand(5, 3), columns=["A", "B", "C"])
    dataframes = {"test": df}
    normalized_data = normalize_data(dataframes, method="standard")
    
    assert np.isclose(normalized_data["test"].mean().mean(), 0, atol=1e-1), "Normalization failed."


if __name__ == "__main__":
    pytest.main()
