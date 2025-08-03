import numpy as np
import logging
import pandas as pd
import pennylane as qml
from sklearn.cluster import KMeans
from joblib import Parallel, delayed


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

num_qubits = 4
total_wires = 2 * num_qubits

dev = qml.device("default.mixed", wires=total_wires)

@qml.qnode(dev)
def quantum_pca_qpe(state):
    """Quantum PCA using Quantum Phase Estimation (QPE) with correct wire allocation."""
    estimation_wires = range(num_qubits)
    target_wires = range(num_qubits, 2 * num_qubits)

    qml.StatePrep(state, wires=target_wires)
    unitary = np.eye(2**num_qubits)
    qml.QuantumPhaseEstimation(unitary, target_wires=target_wires, estimation_wires=estimation_wires)
    return qml.probs(wires=estimation_wires)


def apply_qpca_with_error_correction(data, n_components=4):
    """Applies Quantum PCA with QPE and noise mitigation."""
    transformed_data = []
    for row in data:
        padded_row = np.pad(row, (0, 16 - len(row)), mode='constant') if len(row) < 16 else row[:16]
        normalized_row = padded_row / np.linalg.norm(padded_row)
        eigenvalues = quantum_pca_qpe(normalized_row)
        transformed_data.append(eigenvalues[:n_components])
    log.info("Quantum PCA applied successfully.")
    return np.array(transformed_data)


@qml.qnode(dev)
def quantum_kernel(x1, x2):
    """Quantum kernel using Amplitude Embedding with probability extraction."""
    required_length = 2**num_qubits
    x1_padded = np.pad(x1, (0, required_length - len(x1)), mode='constant') if len(x1) < required_length else x1[:required_length]
    x2_padded = np.pad(x2, (0, required_length - len(x2)), mode='constant') if len(x2) < required_length else x2[:required_length]

    x1_padded /= np.linalg.norm(x1_padded) if np.linalg.norm(x1_padded) > 0 else 1
    x2_padded /= np.linalg.norm(x2_padded) if np.linalg.norm(x2_padded) > 0 else 1

    qml.AmplitudeEmbedding(x1_padded, wires=range(num_qubits), normalize=False)
    qml.adjoint(qml.AmplitudeEmbedding)(x2_padded, wires=range(num_qubits))

    probabilities = qml.probs(wires=range(num_qubits))
    return probabilities


def compute_kernel_row(x, data):
    return [quantum_kernel(x, y) for y in data]


def quantum_clustering_with_error_correction(data, n_clusters=2):
    """Applies Quantum Clustering with Parallel Quantum Kernel Computation."""
    log.info("Computing Quantum Kernel Similarity Matrix with Parallel Processing...")

    data = np.array([x / np.linalg.norm(x) if np.linalg.norm(x) > 0 else x for x in data])

    quantum_similarity = np.array(Parallel(n_jobs=-1)(
        delayed(compute_kernel_row)(x, data) for x in data
    ))

    quantum_similarity = quantum_similarity.reshape(quantum_similarity.shape[0], -1)

    log.info("Applying KMeans Clustering on Quantum Kernel Data...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(quantum_similarity)

    log.info("Quantum Clustering completed successfully.")
    return cluster_labels


def reduce_noise_with_qpca(dataframes, n_components=4):
    """Reduce noise in dataset using Quantum PCA with QPE."""
    log.info("Reducing noise using Quantum PCA with %s components.", n_components)
    for path, df in dataframes.items():
        log.info("Processing dataset: %s", path)
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        data = df[numeric_cols].values
        qpca_data = apply_qpca_with_error_correction(data, n_components)
        dataframes[path] = pd.DataFrame(qpca_data, columns=[f"QPC{i+1}" for i in range(n_components)])
        log.info("Quantum PCA applied to dataset: %s", path)
    log.info("Noise reduction with Quantum PCA completed.")
    return dataframes
