import os
import numpy as np
import pandas as pd
import torch
import logging
from sklearn.model_selection import train_test_split
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_algorithms.optimizers import COBYLA
from qiskit.circuit.library import ZFeatureMap


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = "../models/NeuralNetworkClassifier_treatment_model_qnn.model"


def conv_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def conv_layer(num_qubits, param_prefix):
    qc = QuantumCircuit(num_qubits, name="Convolutional Layer")
    qubits = list(range(num_qubits))
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits * 3)

    for q1, q2 in zip(qubits[0::2], qubits[1::2]):
        qc = qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
        qc.barrier()
        param_index += 3

    if num_qubits % 2 == 0:
        for q1, q2 in zip(qubits[1::2], qubits[2::2] + [0]):
            qc = qc.compose(conv_circuit(params[param_index: param_index + 3]), [q1, q2])
            qc.barrier()
            param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, qubits)
    return qc


def pool_circuit(params):
    target = QuantumCircuit(2)
    target.rz(-np.pi / 2, 1)
    target.cx(1, 0)
    target.rz(params[0], 0)
    target.ry(params[1], 1)
    target.cx(0, 1)
    target.ry(params[2], 1)
    return target


def pool_layer(sources, sinks, param_prefix):
    num_qubits = len(sources) + len(sinks)
    qc = QuantumCircuit(num_qubits, name="Pooling Layer")
    param_index = 0
    params = ParameterVector(param_prefix, length=num_qubits // 2 * 3)

    for source, sink in zip(sources, sinks):
        qc = qc.compose(pool_circuit(params[param_index: param_index + 3]), [source, sink])
        qc.barrier()
        param_index += 3

    qc_inst = qc.to_instruction()
    qc = QuantumCircuit(num_qubits)
    qc.append(qc_inst, range(num_qubits))
    return qc


def create_qnn(num_qubits=8):
    feature_map = ZFeatureMap(num_qubits)
    ansatz = QuantumCircuit(num_qubits, name="Ansatz")

    ansatz.compose(conv_layer(8, "c1"), list(range(8)), inplace=True)
    ansatz.compose(pool_layer([0, 1, 2, 3], [4, 5, 6, 7], "p1"), list(range(8)), inplace=True)
    ansatz.compose(conv_layer(4, "c2"), list(range(4, 8)), inplace=True)
    ansatz.compose(pool_layer([0, 1], [2, 3], "p2"), list(range(4, 8)), inplace=True)
    ansatz.compose(conv_layer(2, "c3"), list(range(6, 8)), inplace=True)
    ansatz.compose(pool_layer([0], [1], "p3"), list(range(6, 8)), inplace=True)

    circuit = QuantumCircuit(8)
    circuit.compose(feature_map, range(8), inplace=True)
    circuit.compose(ansatz, range(8), inplace=True)

    observable = SparsePauliOp.from_list([("Z" + "I" * 7, 1)])
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        observables=observable,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    return qnn


qnn = create_qnn()


def train_quantum_model(X_train, y_train, data_path, qnn):
    num_qubits = 8

    log.info("Loading dataset from %s", data_path)
    data = pd.read_csv(data_path)

    if "QuantumCluster" not in data.columns:
        raise KeyError("Missing 'QuantumCluster' column in dataset.")

    X = data.drop(columns=["QuantumCluster"])
    y = data["QuantumCluster"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_res = X_train.iloc[:, :num_qubits].astype(np.float32).to_numpy()
    y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32)

    if X_train_res.ndim == 1:
        X_train_res = X_train_res.reshape(-1, num_qubits)

    if os.path.exists(MODEL_PATH):
        log.info("Loading pre-trained Quantum Neural Network model from %s", MODEL_PATH)
        classifier = NeuralNetworkClassifier.load(MODEL_PATH)
    else:
        log.info("Training a new Quantum Neural Network model...")
        classifier = NeuralNetworkClassifier(
            neural_network=qnn,
            optimizer=COBYLA(maxiter=200),
            loss="cross_entropy"
        )
        classifier.fit(X_train_res, y_train_tensor)

        classifier.save(MODEL_PATH)
        log.info("Model saved to %s", MODEL_PATH)

    return classifier
