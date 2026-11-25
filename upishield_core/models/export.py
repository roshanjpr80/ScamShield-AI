"""
PyTorch to ONNX Export and Quantization Utilities.

Handles: export, dynamic quantization, static quantization, optimization.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import (
    QuantType,
    quantize_dynamic,
    quantize_static,
    CalibrationDataReader,
)
from onnxruntime.quantization.shape_inference import quant_pre_process


class OnnxDemoDataReader(CalibrationDataReader):
    """Calibration data reader for static quantization."""

    def __init__(self, input_name: str, calibration_data: np.ndarray):
        self.input_name = input_name
        self.data = calibration_data
        self.idx = 0
        self.batch_size = 1

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self.idx >= len(self.data):
            return None
        batch = self.data[self.idx:self.idx + self.batch_size]
        self.idx += self.batch_size
        return {self.input_name: batch.astype(np.float32)}

    def rewind(self) -> None:
        self.idx = 0


def export_to_onnx(
    model: nn.Module,
    output_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 50),
    input_names: List[str] = None,
    output_names: List[str] = None,
    dynamic_axes: Dict[str, Dict[int, str]] = None,
    opset_version: int = 17,
    do_constant_folding: bool = True,
    verbose: bool = False,
) -> str:
    """
    Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model (must be in eval mode)
        output_path: Path to save ONNX model
        input_shape: Shape of dummy input (batch_size, features)
        input_names: Names for input tensors
        output_names: Names for output tensors
        dynamic_axes: Dynamic axes for variable batch size
        opset_version: ONNX opset version
        do_constant_folding: Optimize constants
        verbose: Print export details

    Returns:
        Path to exported model
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Default names
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    # Default dynamic axes (variable batch size)
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }

    # Create dummy input
    dummy_input = torch.randn(*input_shape)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=do_constant_folding,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=verbose,
    )

    # Validate
    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)

    print(f"Exported to {output_path} (opset {opset_version})")
    return str(output_path)


def quantize_dynamic_int8(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = True,
    reduce_range: bool = False,
    extra_options: Dict = None,
) -> str:
    """
    Dynamic quantization (weights only, activations stay FP32).

    Best for: XGBoost, LightGBM, linear models
    Speedup: 2-3x, Accuracy drop: <0.5%
    """
    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    extra_options = extra_options or {
        "EnableSubgraph": True,
        "ForceQuantizeNoOpCheck": True,
    }

    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        extra_options=extra_options,
    )

    # Validate quantized model
    _validate_quantized_model(input_path, output_path)

    print(f"Dynamic INT8 quantization: {input_path} -> {output_path}")
    return output_path


def quantize_static_int8(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    calibration_data: np.ndarray,
    input_name: str = "input",
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = True,
    reduce_range: bool = True,
    calibrate_method: str = "minmax",
) -> str:
    """
    Static quantization (weights + activations quantized).

    Best for: Transformers, CNNs, models with activation outliers
    Speedup: 3-4x, Accuracy drop: <1%
    Requires: Calibration dataset
    """
    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Pre-process for better quantization
    preprocessed_path = str(Path(output_path).parent / f"{Path(output_path).stem}_pre.onnx")
    quant_pre_process(input_path, preprocessed_path)

    # Create calibration reader
    reader = OnnxDemoDataReader(input_name, calibration_data)

    quantize_static(
        model_input=preprocessed_path,
        model_output=output_path,
        calibration_data_reader=reader,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
        reduce_range=reduce_range,
        calibrate_method=calibrate_method,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
            "EnableSubgraph": True,
        },
    )

    # Cleanup
    os.remove(preprocessed_path)

    _validate_quantized_model(input_path, output_path)

    print(f"Static INT8 quantization: {input_path} -> {output_path}")
    return output_path


def optimize_onnx(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    model_type: str = "bert",
    num_heads: int = 12,
    hidden_size: int = 768,
    optimization_level: int = 99,
) -> str:
    """
    Optimize ONNX model using ONNX Runtime transformer optimizer.

    Applies: layer fusion, constant folding, layout optimization, etc.
    """
    input_path = str(input_path)
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    from onnxruntime.transformers import optimizer

    opt_model = optimizer.optimize_model(
        input_path,
        model_type=model_type,
        num_heads=num_heads,
        hidden_size=hidden_size,
        opt_level=optimization_level,
    )
    opt_model.save_model_to_file(output_path)

    print(f"Optimized: {input_path} -> {output_path}")
    return output_path


def _validate_quantized_model(original_path: str, quantized_path: str) -> None:
    """Validate quantized model loads and runs."""
    # Check it loads
    sess = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])

    # Run inference
    input_name = sess.get_inputs()[0].name
    dummy = np.random.randn(1, 50).astype(np.float32)
    _ = sess.run(None, {input_name: dummy})

    # Compare sizes
    orig_size = os.path.getsize(original_path)
    quant_size = os.path.getsize(quantized_path)
    reduction = (1 - quant_size / orig_size) * 100

    print(f"  Original: {orig_size / 1024 / 1024:.2f} MB")
    print(f"  Quantized: {quant_size / 1024 / 1024:.2f} MB")
    print(f"  Reduction: {reduction:.1f}%")


def benchmark_model(
    model_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 50),
    num_runs: int = 1000,
    warmup_runs: int = 100,
    num_threads: int = 1,
) -> Dict[str, float]:
    """
    Benchmark ONNX model latency.

    Returns: dict with p50, p95, p99, mean, std in milliseconds
    """
    sess = ort.InferenceSession(
        str(model_path),
        sess_options=ort.SessionOptions(),
        providers=["CPUExecutionProvider"],
    )
    sess._sess_options.intra_op_num_threads = num_threads

    input_name = sess.get_inputs()[0].name

    # Warmup
    dummy = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(warmup_runs):
        _ = sess.run(None, {input_name: dummy})

    # Benchmark
    latencies = []
    for _ in range(num_runs):
        dummy = np.random.randn(*input_shape).astype(np.float32)
        import time
        start = time.perf_counter()
        _ = sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000)

    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
    }


def compare_models(
    original_path: Union[str, Path],
    quantized_path: Union[str, Path],
    input_shape: Tuple[int, ...] = (1, 50),
    num_samples: int = 1000,
) -> Dict[str, Any]:
    """
    Compare accuracy and latency between original and quantized models.
    """
    orig_sess = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])
    quant_sess = ort.InferenceSession(str(quantized_path), providers=["CPUExecutionProvider"])

    input_name = orig_sess.get_inputs()[0].name

    # Generate test data
    np.random.seed(42)
    test_inputs = [np.random.randn(*input_shape).astype(np.float32) for _ in range(num_samples)]

    # Run both
    orig_outputs = []
    quant_outputs = []
    orig_latencies = []
    quant_latencies = []

    import time
    for x in test_inputs:
        # Original
        start = time.perf_counter()
        o = orig_sess.run(None, {input_name: x})
        orig_latencies.append((time.perf_counter() - start) * 1000)
        orig_outputs.append(o[0])

        # Quantized
        start = time.perf_counter()
        q = quant_sess.run(None, {input_name: x})
        quant_latencies.append((time.perf_counter() - start) * 1000)
        quant_outputs.append(q[0])

    orig_outputs = np.array(orig_outputs)
    quant_outputs = np.array(quant_outputs)

    # Accuracy metrics
    mse = np.mean((orig_outputs - quant_outputs) ** 2)
    mae = np.mean(np.abs(orig_outputs - quant_outputs))
    max_diff = np.max(np.abs(orig_outputs - quant_outputs))
    cos_sim = np.mean([
        np.dot(o.flatten(), q.flatten()) / (np.linalg.norm(o) * np.linalg.norm(q) + 1e-8)
        for o, q in zip(orig_outputs, quant_outputs)
    ])

    return {
        "accuracy": {
            "mse": float(mse),
            "mae": float(mae),
            "max_diff": float(max_diff),
            "cosine_similarity": float(cos_sim),
        },
        "latency_original": {
            "mean_ms": float(np.mean(orig_latencies)),
            "p50_ms": float(np.percentile(orig_latencies, 50)),
            "p95_ms": float(np.percentile(orig_latencies, 95)),
            "p99_ms": float(np.percentile(orig_latencies, 99)),
        },
        "latency_quantized": {
            "mean_ms": float(np.mean(quant_latencies)),
            "p50_ms": float(np.percentile(quant_latencies, 50)),
            "p95_ms": float(np.percentile(quant_latencies, 95)),
            "p99_ms": float(np.percentile(quant_latencies, 99)),
        },
        "speedup": float(np.mean(orig_latencies) / np.mean(quant_latencies)),
        "size_reduction_pct": float((1 - os.path.getsize(str(quantized_path)) / os.path.getsize(str(original_path))) * 100),
    }


def export_pipeline(
    pytorch_model: nn.Module,
    output_dir: Union[str, Path],
    model_name: str,
    calibration_data: Optional[np.ndarray] = None,
    quantization: str = "dynamic",  # "dynamic", "static", "none"
    optimize: bool = True,
) -> Dict[str, str]:
    """
    Complete export pipeline: PyTorch -> ONNX -> Quantize -> Optimize.

    Returns dict with paths to all generated models.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. Export FP32 ONNX
    fp32_path = output_dir / f"{model_name}.onnx"
    export_to_onnx(pytorch_model, fp32_path)
    paths["fp32"] = str(fp32_path)

    # 2. Quantize
    if quantization == "dynamic":
        int8_path = output_dir / f"{model_name}_int8.onnx"
        quantize_dynamic_int8(fp32_path, int8_path)
        paths["int8"] = str(int8_path)
    elif quantization == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        int8_path = output_dir / f"{model_name}_int8.onnx"
        quantize_static_int8(fp32_path, int8_path, calibration_data)
        paths["int8"] = str(int8_path)
    else:
        int8_path = fp32_path

    # 3. Optimize (for transformer models)
    if optimize and "transformer" in model_name.lower():
        opt_path = output_dir / f"{model_name}_opt.onnx"
        optimize_onnx(int8_path, opt_path)
        paths["optimized"] = str(opt_path)

    # 4. Benchmark all
    print("\n=== Benchmark Results ===")
    for key, path in paths.items():
        bench = benchmark_model(path)
        print(f"{key}: p50={bench['p50_ms']:.2f}ms p95={bench['p95_ms']:.2f}ms p99={bench['p99_ms']:.2f}ms")

    # 5. Compare if quantized
    if "int8" in paths and "int8" != "fp32":
        print("\n=== Accuracy Comparison ===")
        cmp = compare_models(paths["fp32"], paths["int8"])
        print(f"  Cosine Similarity: {cmp['accuracy']['cosine_similarity']:.6f}")
        print(f"  MSE: {cmp['accuracy']['mse']:.6f}")
        print(f"  Speedup: {cmp['speedup']:.2f}x")
        print(f"  Size Reduction: {cmp['size_reduction_pct']:.1f}%")

    return paths