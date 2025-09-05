import os
import sys
import logging
import numpy as np
import torch
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier
from qiskit_aer import AerSimulator
from qiskit.circuit.library import IQP
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit import transpile
from qiskit_ibm_runtime import Session

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(BASE_DIR, "../."))
from workflow.job_scheduler import JobScheduler
from treatment_recommendation.recommendation_engine import TreatmentRecommendation

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

MODEL_PATH = "../models/NeuralNetworkClassifier_treatment_model_qnn.model"

token = os.getenv("IBM_API")

if not token:
    raise ValueError("ERROR: IBM_API environment variable is not set!")

QiskitRuntimeService.save_account(token=token, channel="ibm_cloud", overwrite=True)
service = QiskitRuntimeService(channel="ibm_cloud")

backend = service.least_busy(operational=True, simulator=False)

class WorkflowManager:
    def __init__(self, integrated_data, label_encoder):
        """Initialize Workflow Manager with Quantum AI-Powered Drug Discovery"""
        self.integrated_data = integrated_data
        self.label_encoder = label_encoder
        self.job_scheduler = JobScheduler()
        self.treatment_recommendation = TreatmentRecommendation(integrated_data, label_encoder)

        log.info("Quantum AI Workflow Initialized on Backend: %s", backend)

    def train_hybrid_model(self):
        """Train the Hybrid Quantum-Classical Model with Job Scheduling"""
        log.info("Scheduling Model Training Jobs...")
        self.job_scheduler.schedule_task(self._execute_training)

    def _execute_training(self):
        """Handles Quantum Training Execution"""
        log.info("Executing Hybrid Quantum-Classical Model Training...")

        # Step 1: Train the Quantum Neural Network Model
        self.treatment_recommendation.train_model()

        # Step 2: Quantum Circuit Optimization
        log.info("Generating Quantum Circuit for Optimization...")
        num_qubits = 8
        mat = np.real(np.random.rand(num_qubits, num_qubits))
        circuit = IQP(mat)

        observable = SparsePauliOp("Z" * num_qubits)

        pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
        optimized_circuit = pm.run(circuit)
        optimized_observable = observable.apply_layout(optimized_circuit.layout)

        # Step 3: Run Quantum Job via IBM Quantum
        log.info("Executing Quantum Job on IBM Backend...")
        
        with Session(backend=backend) as session:
            estimator = Estimator(mode=session)
            job = estimator.run([(optimized_circuit, optimized_observable)])
            result = job.result()

        log.info("Quantum Execution Completed. Expectation Value: %s", result[0].data.evs)

    def infer_treatment(self, patient_features):
        """Infer Treatment using Trained Hybrid Model"""
        log.info("Scheduling Inference Task...")
        return self.job_scheduler.schedule_task(self.treatment_recommendation.recommend_treatment, patient_features)