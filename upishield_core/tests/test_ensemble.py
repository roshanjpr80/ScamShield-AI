"""Tests for ONNX Ensemble Inference."""

from unittest.mock import MagicMock

import numpy as np
import pytest


class TestEnsembleInference:
    """Test EnsembleInference class."""

    def test_ensemble_initialization(self):
        """Test ensemble initializes with correct defaults."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference(model_dir="./models")
        assert str(ensemble.model_dir) == "models"
        assert ensemble.weights == ensemble.DEFAULT_WEIGHTS
        assert not ensemble._loaded

    def test_ensemble_custom_weights(self):
        """Test ensemble with custom weights."""
        from upishield_core.models.ensemble import EnsembleInference

        custom_weights = {"xgb": 0.4, "lgbm": 0.3, "transformer": 0.2, "gnn": 0.1}
        ensemble = EnsembleInference(model_dir="./models", weights=custom_weights)
        assert ensemble.weights == custom_weights

    def test_prepare_input(self, sample_transaction):
        """Test input preparation from transaction and features."""
        from datetime import datetime

        from upishield_core.features.schemas import UPIFeatures
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        features = UPIFeatures(
            txn_id="TEST",
            payer_vpa="user@upi",
            payee_vpa="merchant@upi",
            amount=5000.0,
            timestamp=datetime.utcnow(),
        )

        input_tensor = ensemble._prepare_input(sample_transaction, features)

        assert input_tensor.shape == (1, 13)
        assert input_tensor.dtype == np.float32

    def test_prepare_input_without_features(self, sample_transaction):
        """Test input preparation with only transaction."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        input_tensor = ensemble._prepare_input(sample_transaction, None)

        assert input_tensor.shape == (1, 13)
        assert input_tensor.dtype == np.float32

    @pytest.mark.asyncio
    async def test_predict_async_not_loaded(self, sample_transaction):
        """Test predict_async loads models if not loaded."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()

        # Mock load and sessions
        ensemble.load = MagicMock()
        ensemble.sessions = {"xgb": MagicMock()}
        ensemble._run_model = MagicMock(
            return_value=MagicMock(
                model_name="xgb",
                risk_score=0.5,
                confidence=0.8,
                latency_ms=5.0,
                raw_output=np.array([[0.5, 0.5]]),
            )
        )
        ensemble._loaded = False

        predictions = await ensemble.predict_async(sample_transaction)

        assert ensemble.load.called
        assert len(predictions) == 1

    def test_ensemble_score(self):
        """Test ensemble score computation."""
        import numpy as np

        from upishield_core.models.ensemble import EnsembleInference, ModelPrediction

        ensemble = EnsembleInference()
        predictions = [
            ModelPrediction("xgb", 0.8, 0.9, 5.0, np.array([[0.2, 0.8]])),
            ModelPrediction("lgbm", 0.7, 0.85, 4.0, np.array([[0.3, 0.7]])),
            ModelPrediction("transformer", 0.75, 0.88, 6.0, np.array([[0.25, 0.75]])),
            ModelPrediction("gnn", 0.6, 0.8, 7.0, np.array([[0.4, 0.6]])),
        ]

        score, metadata = ensemble.ensemble_score(predictions)

        assert 0.0 <= score <= 1.0
        assert "uncertainty" in metadata
        assert "agreement" in metadata
        assert "individual_scores" in metadata
        assert metadata["individual_scores"]["xgb"] == 0.8

    def test_ensemble_score_empty(self):
        """Test ensemble score with empty predictions."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        score, metadata = ensemble.ensemble_score([])

        assert score == 0.0

    def test_predict_with_uncertainty(self, sample_transaction):
        """Test predict_with_uncertainty method."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        ensemble._loaded = True
        ensemble.sessions = {"xgb": MagicMock(), "lgbm": MagicMock()}

        # Mock _run_model
        ensemble._run_model = MagicMock(
            side_effect=[
                MagicMock(
                    model_name="xgb",
                    risk_score=0.8,
                    confidence=0.9,
                    latency_ms=5.0,
                    raw_output=np.array([[0.8]]),
                ),
                MagicMock(
                    model_name="lgbm",
                    risk_score=0.7,
                    confidence=0.85,
                    latency_ms=4.0,
                    raw_output=np.array([[0.7]]),
                ),
            ]
        )

        score, confidence, model_scores = ensemble.predict_with_uncertainty(sample_transaction)

        assert 0.0 <= score <= 1.0
        assert 0.0 <= confidence <= 1.0
        assert "xgb" in model_scores
        assert "lgbm" in model_scores

    def test_predict_with_uncertainty_1d_input(self, sample_transaction):
        """Test predict_with_uncertainty with transaction object."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        ensemble._loaded = True
        ensemble.sessions = {"xgb": MagicMock()}

        ensemble._run_model = MagicMock(
            return_value=MagicMock(
                model_name="xgb",
                risk_score=0.75,
                confidence=0.85,
                latency_ms=5.0,
                raw_output=np.array([[0.75]]),
            )
        )

        score, confidence, model_scores = ensemble.predict_with_uncertainty(sample_transaction)

        assert 0.0 <= score <= 1.0

    def test_context_manager(self):
        """Test ensemble as context manager."""
        from upishield_core.models.ensemble import EnsembleInference

        ensemble = EnsembleInference()
        ensemble.load = MagicMock()
        ensemble.executor.shutdown = MagicMock()

        with ensemble as e:
            assert e is ensemble

        ensemble.executor.shutdown.assert_called_once_with(wait=True)


class TestONNXEnsembleAlias:
    """Test ONNXEnsemble alias."""

    def test_alias_works(self):
        """Test ONNXEnsemble is alias for EnsembleInference."""
        from upishield_core.models.ensemble import EnsembleInference, ONNXEnsemble

        assert ONNXEnsemble is EnsembleInference


class TestModelPrediction:
    """Test ModelPrediction dataclass."""

    def test_model_prediction_creation(self):
        """Test creating ModelPrediction."""
        import numpy as np

        from upishield_core.models.ensemble import ModelPrediction

        pred = ModelPrediction(
            model_name="xgb",
            risk_score=0.75,
            confidence=0.85,
            latency_ms=5.0,
            raw_output=np.array([[0.25, 0.75]]),
        )

        assert pred.model_name == "xgb"
        assert pred.risk_score == 0.75
        assert pred.confidence == 0.85
        assert pred.latency_ms == 5.0
