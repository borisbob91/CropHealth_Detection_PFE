#!/usr/bin/env python3
"""
CropHealth Detection - Benchmark Exported Models
Compare PyTorch, ONNX, TensorRT, TFLite performance

Usage:
    python export/benchmark_exports.py --model ssd \
                                       --pytorch runs/CropHealth_SSD/best.pt \
                                       --onnx exports/ssd/CropHealth_SSD.onnx \
                                       --tensorrt exports/ssd/CropHealth_SSD.engine \
                                       --tflite exports/ssd/CropHealth_SSD_int8.tflite \
                                       --input data/test/images \
                                       --output benchmark_results.csv
"""
import argparse
import csv
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config.model_configs import MODEL_CONFIGS, NUM_CLASSES
from models.ssd_model import build_ssd_model
from models.effdet_model import build_efficientdet_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model


def benchmark_pytorch(model_path, model_key, test_images, num_runs=100):
    """Benchmark PyTorch model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking PyTorch...")
    print(f"{'='*60}\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = MODEL_CONFIGS[model_key]
    input_size = config['input_size']
    
    # Load model
    if model_key == 'ssd':
        model = build_ssd_model(NUM_CLASSES)
    elif model_key == 'efficientdet':
        model = build_efficientdet_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn':
        model = build_fasterrcnn_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn_light':
        model = build_fasterrcnn_light_model(NUM_CLASSES)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Prepare test images
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model([dummy_input])
    
    # Benchmark
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model([dummy_input])
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"✅ PyTorch - Avg time: {avg_time:.2f} ms | FPS: {fps:.2f}")
    
    return {
        'backend': 'PyTorch',
        'avg_time_ms': avg_time,
        'fps': fps,
        'device': str(device)
    }


def benchmark_onnx(onnx_path, input_size, num_runs=100):
    """Benchmark ONNX model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking ONNX Runtime...")
    print(f"{'='*60}\n")
    
    try:
        import onnxruntime as ort
    except ImportError:
        print(f"❌ onnxruntime not installed")
        return None
    
    # Session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Input
    input_name = session.get_inputs()[0].name
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        _ = session.run(None, {input_name: dummy_input})
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    fps = 1000 / avg_time
    
    device = 'CUDA' if 'CUDAExecutionProvider' in session.get_providers() else 'CPU'
    
    print(f"✅ ONNX - Avg time: {avg_time:.2f} ms | FPS: {fps:.2f}")
    
    return {
        'backend': 'ONNX Runtime',
        'avg_time_ms': avg_time,
        'fps': fps,
        'device': device
    }


def benchmark_tensorrt(engine_path, input_size, num_runs=100):
    """Benchmark TensorRT engine"""
    print(f"\n{'='*60}")
    print(f"Benchmarking TensorRT...")
    print(f"{'='*60}\n")
    
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print(f"❌ TensorRT or PyCUDA not installed")
        return None
    
    # Logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Load engine
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers
    h_input = cuda.pagelocked_empty(trt.volume((1, 3, input_size, input_size)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Warmup
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"✅ TensorRT - Avg time: {avg_time:.2f} ms | FPS: {fps:.2f}")
    
    return {
        'backend': 'TensorRT',
        'avg_time_ms': avg_time,
        'fps': fps,
        'device': 'CUDA'
    }


def benchmark_tflite(tflite_path, input_size, num_runs=100):
    """Benchmark TFLite model"""
    print(f"\n{'='*60}")
    print(f"Benchmarking TFLite...")
    print(f"{'='*60}\n")
    
    try:
        import tensorflow as tf
    except ImportError:
        print(f"❌ TensorFlow not installed")
        return None
    
    # Interpreter
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Input
    input_shape = input_details[0]['shape']
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times.append(time.perf_counter() - start)
    
    avg_time = np.mean(times) * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"✅ TFLite - Avg time: {avg_time:.2f} ms | FPS: {fps:.2f}")
    
    return {
        'backend': 'TFLite',
        'avg_time_ms': avg_time,
        'fps': fps,
        'device': 'CPU'
    }


def save_benchmark_csv(results, output_path):
    """Save benchmark results to CSV"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Backend', 'Avg Time (ms)', 'FPS', 'Device', 'Speedup'])
        
        # Calculate speedup (vs PyTorch)
        pytorch_time = next((r['avg_time_ms'] for r in results if r['backend'] == 'PyTorch'), None)
        
        for result in results:
            speedup = pytorch_time / result['avg_time_ms'] if pytorch_time else 1.0
            writer.writerow([
                result['backend'],
                f"{result['avg_time_ms']:.2f}",
                f"{result['fps']:.2f}",
                result['device'],
                f"{speedup:.2f}x"
            ])
    
    print(f"\n✅ Benchmark results saved: {output_path}")


def main(args):
    config = MODEL_CONFIGS[args.model]
    input_size = config['input_size']
    
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Export Benchmark")
    print(f"Model: {args.model}")
    print(f"Input size: {input_size}x{input_size}")
    print(f"Runs: {args.runs}")
    print(f"{'='*60}")
    
    results = []
    
    # PyTorch
    if args.pytorch:
        result = benchmark_pytorch(args.pytorch, args.model, args.input, args.runs)
        if result:
            results.append(result)
    
    # ONNX
    if args.onnx:
        result = benchmark_onnx(args.onnx, input_size, args.runs)
        if result:
            results.append(result)
    
    # TensorRT
    if args.tensorrt:
        result = benchmark_tensorrt(args.tensorrt, input_size, args.runs)
        if result:
            results.append(result)
    
    # TFLite
    if args.tflite:
        result = benchmark_tflite(args.tflite, input_size, args.runs)
        if result:
            results.append(result)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"📊 BENCHMARK SUMMARY")
    print(f"{'='*60}\n")
    print(f"{'Backend':<20} {'Time (ms)':<15} {'FPS':<10} {'Device':<10}")
    print(f"{'-'*55}")
    
    for result in results:
        print(f"{result['backend']:<20} {result['avg_time_ms']:<15.2f} {result['fps']:<10.2f} {result['device']:<10}")
    
    # Save CSV
    if args.output:
        save_benchmark_csv(results, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropHealth Export Benchmark',
                                     usage=__doc__)
    parser.add_argument('--model', type=str, required=True,
                        choices=['ssd', 'efficientdet', 'fasterrcnn', 'fasterrcnn_light'],
                        help='Model architecture')
    parser.add_argument('--pytorch', type=str,
                        help='PyTorch checkpoint path')
    parser.add_argument('--onnx', type=str,
                        help='ONNX model path')
    parser.add_argument('--tensorrt', type=str,
                        help='TensorRT engine path')
    parser.add_argument('--tflite', type=str,
                        help='TFLite model path')
    parser.add_argument('--input', type=str,
                        help='Test images directory (optional)')
    parser.add_argument('--runs', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--output', type=str, default='benchmark_results.csv',
                        help='Output CSV path')
    
    args = parser.parse_args()
    
    main(args)