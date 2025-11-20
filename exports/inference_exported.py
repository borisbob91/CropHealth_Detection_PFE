#!/usr/bin/env python3
"""
CropHealth Detection - Inference on Exported Models
Inférence rapide sur ONNX, TensorRT, TFLite

Usage:
    # ONNX
    python export/inference_exported.py --model exports/ssd/CropHealth_SSD.onnx \
                                        --format onnx \
                                        --input data/test/image.jpg \
                                        --output predictions/

    # TensorRT
    python export/inference_exported.py --model exports/ssd/CropHealth_SSD.engine \
                                        --format tensorrt \
                                        --input data/test/image.jpg

    # TFLite
    python export/inference_exported.py --model exports/ssd/CropHealth_SSD_int8.tflite \
                                        --format tflite \
                                        --input data/test/image.jpg
"""
import argparse
import time
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw


class ONNXInference:
    """ONNX Runtime inference"""
    def __init__(self, model_path):
        import onnxruntime as ort
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
    
    def preprocess(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[3]))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        img = np.expand_dims(img, axis=0)
        return img
    
    def __call__(self, img_tensor):
        outputs = self.session.run(None, {self.input_name: img_tensor})
        # outputs = [boxes, labels, scores] ou format spécifique modèle
        return outputs


class TensorRTInference:
    """TensorRT inference"""
    def __init__(self, engine_path):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        self.input_shape = self.engine.get_binding_shape(0)
        
        # Allocate buffers
        self.h_input = cuda.pagelocked_empty(trt.volume(self.input_shape), dtype=np.float32)
        self.h_output = cuda.pagelocked_empty(trt.volume(self.engine.get_binding_shape(1)), dtype=np.float32)
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream()
    
    def preprocess(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[2], self.input_shape[3]))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = img.flatten()
        return img
    
    def __call__(self, img_tensor):
        import pycuda.driver as cuda
        
        np.copyto(self.h_input, img_tensor)
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
        self.context.execute_async_v2(bindings=[int(self.d_input), int(self.d_output)], 
                                       stream_handle=self.stream.handle)
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
        self.stream.synchronize()
        return [self.h_output]


class TFLiteInference:
    """TFLite inference"""
    def __init__(self, model_path):
        import tensorflow as tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
    
    def preprocess(self, img_path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[2]))
        
        # INT8 quantization
        if self.input_details[0]['dtype'] == np.uint8:
            img = img.astype(np.uint8)
        else:
            img = img.astype(np.float32) / 255.0
        
        img = np.expand_dims(img, axis=0)
        return img
    
    def __call__(self, img_tensor):
        self.interpreter.set_tensor(self.input_details[0]['index'], img_tensor)
        self.interpreter.invoke()
        
        outputs = []
        for output_detail in self.output_details:
            outputs.append(self.interpreter.get_tensor(output_detail['index']))
        return outputs


def visualize_detections(img_path, boxes, labels, scores, output_path, conf_threshold=0.5):
    """Visualise détections sur image"""
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    if isinstance(boxes, np.ndarray):
        boxes = boxes.tolist() if boxes.ndim > 1 else []
        labels = labels.tolist() if hasattr(labels, 'tolist') else labels
        scores = scores.tolist() if hasattr(scores, 'tolist') else scores
    
    num_detections = 0
    for box, label, score in zip(boxes, labels, scores):
        if score < conf_threshold:
            continue
        
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        text = f"Class {int(label)}: {score:.2f}"
        draw.text((x1, y1 - 10), text, fill='red')
        num_detections += 1
    
    img.save(output_path)
    return num_detections


def main(args):
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Exported Model Inference")
    print(f"Model: {args.model}")
    print(f"Format: {args.format.upper()}")
    print(f"{'='*60}\n")
    
    # Load model
    print(f"📦 Loading {args.format.upper()} model...")
    
    if args.format == 'onnx':
        model = ONNXInference(args.model)
    elif args.format == 'tensorrt':
        model = TensorRTInference(args.model)
    elif args.format == 'tflite':
        model = TFLiteInference(args.model)
    else:
        raise ValueError(f"Unknown format: {args.format}")
    
    print(f"✅ Model loaded")
    
    # Process images
    input_path = Path(args.input)
    
    if input_path.is_file():
        img_paths = [input_path]
    else:
        img_paths = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
    
    print(f"\n🔍 Processing {len(img_paths)} images...\n")
    
    total_time = 0
    for img_path in img_paths:
        # Preprocess
        img_tensor = model.preprocess(img_path)
        
        # Inference
        start = time.perf_counter()
        outputs = model(img_tensor)
        inference_time = (time.perf_counter() - start) * 1000
        total_time += inference_time
        
        # Parse outputs (format dépend du modèle)
        boxes = outputs[0] if len(outputs) > 0 else np.array([])
        labels = outputs[1] if len(outputs) > 1 else np.array([])
        scores = outputs[2] if len(outputs) > 2 else np.array([])
        
        # Visualize
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / img_path.name
            
            num_detections = visualize_detections(
                img_path, boxes, labels, scores, 
                output_path, args.conf
            )
            
            print(f"✅ {img_path.name}: {num_detections} detections | {inference_time:.2f} ms")
        else:
            print(f"✅ {img_path.name}: {inference_time:.2f} ms")
    
    # Summary
    avg_time = total_time / len(img_paths)
    fps = 1000 / avg_time
    
    print(f"\n{'='*60}")
    print(f"📊 SUMMARY")
    print(f"{'='*60}")
    print(f"  Images processed: {len(img_paths)}")
    print(f"  Avg inference time: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total time: {total_time:.2f} ms")
    
    if args.output:
        print(f"  Output: {args.output}")
    
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference on Exported Models', usage=__doc__)
    parser.add_argument('--model', type=str, required=True,
                        help='Exported model path (.onnx / .engine / .tflite)')
    parser.add_argument('--format', type=str, required=True,
                        choices=['onnx', 'tensorrt', 'tflite'],
                        help='Model format')
    parser.add_argument('--input', type=str, required=True,
                        help='Input image or directory')
    parser.add_argument('--output', type=str,
                        help='Output directory for visualizations (optional)')
    parser.add_argument('--conf', type=float, default=0.5,
                        help='Confidence threshold')
    
    args = parser.parse_args()
    main(args)