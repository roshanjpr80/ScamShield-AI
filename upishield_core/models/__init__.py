"""Model inference module."""

from upishield_core.models.ensemble import EnsembleInference, ONNXEnsemble, ModelPrediction
from upishield_core.models.registry import ModelRegistry
from upishield_core.models.export import (
    export_to_onnx,
    quantize_dynamic_int8,
    quantize_static_int8,
    optimize_onnx,
    benchmark_model,
    compare_models,
    export_pipeline,
)
from upishield_core.models.onnx_inference import (
    TextCNNInference,
    JobScamInference,
    SmishingPrediction,
    JobFraudPrediction,
    SmishingCategory,
    JobFraudCategory,
    get_sms_detector,
    get_job_detector,
)

__all__ = [
    "EnsembleInference",
    "ONNXEnsemble",
    "ModelPrediction",
    "ModelRegistry",
    "export_to_onnx",
    "quantize_dynamic_int8",
    "quantize_static_int8",
    "optimize_onnx",
    "benchmark_model",
    "compare_models",
    "export_pipeline",
    "TextCNNInference",
    "JobScamInference",
    "SmishingPrediction",
    "JobFraudPrediction",
    "SmishingCategory",
    "JobFraudCategory",
    "get_sms_detector",
    "get_job_detector",
]