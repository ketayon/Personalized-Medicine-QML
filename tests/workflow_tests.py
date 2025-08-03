import pytest
import numpy as np
import torch
import logging
from unittest.mock import MagicMock, patch
from workflow.workflow_manager import WorkflowManager
from workflow.job_scheduler import JobScheduler
from treatment_recommendation.recommendation_engine import TreatmentRecommendation
from qiskit_ibm_runtime import Session
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# Mock integrated_data and label_encoder for testing
@pytest.fixture
def mock_integrated_data():
    return np.random.rand(10, 8)  # Simulated patient biomarker data


@pytest.fixture
def mock_label_encoder():
    label_encoder = MagicMock()
    label_encoder.inverse_transform.side_effect = lambda x: [f"Treatment_{i}" for i in x]
    return label_encoder


@pytest.fixture
def mock_workflow_manager(mock_integrated_data, mock_label_encoder):
    return WorkflowManager(mock_integrated_data, mock_label_encoder)


@pytest.fixture
def mock_job_scheduler():
    return JobScheduler()


# --- TEST JOB SCHEDULER ---
def test_schedule_task(mock_job_scheduler):
    def sample_task(x):
        return x ** 2
    
    future_result = mock_job_scheduler.schedule_task(sample_task, 5)
    assert future_result == 25
    log.info("Job scheduling test passed.")


# --- TEST WORKFLOW MANAGER ---
def test_workflow_initialization(mock_workflow_manager):
    """Test that WorkflowManager initializes properly with quantum backend."""
    assert mock_workflow_manager is not None
    assert mock_workflow_manager.job_scheduler is not None
    assert mock_workflow_manager.treatment_recommendation is not None
    log.info("WorkflowManager initialized successfully.")


@patch("workflow.workflow_manager.TreatmentRecommendation.train_model")
@patch("workflow.workflow_manager.Session", autospec=True)
@patch("workflow.workflow_manager.Estimator")
def test_train_hybrid_model(mock_estimator, mock_session, mock_train_model, mock_workflow_manager):
    """Test hybrid model training with quantum-classical pipeline."""
    mock_train_model.return_value = None  # Simulate model training

    mock_session_instance = MagicMock()
    mock_session.return_value.__enter__.return_value = mock_session_instance  # Mock session context manager

    mock_estimator_instance = MagicMock()
    mock_estimator.return_value = mock_estimator_instance

    # Simulate job execution returning mock expectation value
    mock_estimator_instance.run.return_value.result.return_value = [MagicMock(data=MagicMock(evs=[0.98]))]

    # Execute training
    mock_workflow_manager.train_hybrid_model()

    # Check that training function is called
    mock_train_model.assert_called_once()
    mock_estimator_instance.run.assert_called_once()
    log.info("Hybrid model training executed successfully.")
