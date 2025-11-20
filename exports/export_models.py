#!/usr/bin/env python3
"""
CropHealth Detection - Model Export
Export models to ONNX, TensorRT, TFLite (INT8), CoreML for mobile/edge deployment

Usage:
    # Export SSD to ONNX
    python export/export_models.py --model ssd \
                                    --checkpoint runs/CropHealth_SSD/best.pt \
                                    --format onnx \
                                    --output exports/ssd

    # Export YOLOv8n to TFLite INT8
    python export/export_models.py --model yolov8n \
                                    --checkpoint runs/CropHealth_YOLOv8n/weights/best.pt \
                                    --format tflite \
                                    --quantize int8 \
                                    --calibration-data data/yolo_crop/train/images \
                                    --output exports/yolov8n

    # Export Faster R-CNN to TensorRT
    python export/export_models.py --model fasterrcnn \
                                    --checkpoint runs/CropHealth_FasterRCNN/best.pt \
                                    --format tensorrt \
                                    --output exports/fasterrcnn
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from config.model_configs import MODEL_CONFIGS, NUM_CLASSES
from models.ssd_model import build_ssd_model
from models.effdet_model import build_efficientdet_model
from models.frcnn_model import build_fasterrcnn_model
from models.frcnn_light_model import build_fasterrcnn_light_model


def build_model(model_key, checkpoint_path, device):
    """Charge le modèle depuis checkpoint"""
    if model_key == 'ssd':
        model = build_ssd_model(NUM_CLASSES)
    elif model_key == 'efficientdet':
        model = build_efficientdet_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn':
        model = build_fasterrcnn_model(NUM_CLASSES)
    elif model_key == 'fasterrcnn_light':
        model = build_fasterrcnn_light_model(NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_key}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    return model


def export_onnx(model, model_key, input_size, output_path, opset_version=14, simplify=True):
    """
    Export to ONNX format
    
    Args:
        model: PyTorch model
        model_key: Model name
        input_size: Input resolution
        output_path: Output .onnx file path
        opset_version: ONNX opset version (default 14 for better compatibility)
        simplify: Simplify ONNX graph (requires onnx-simplifier)
    """
    print(f"\n{'='*60}")
    print(f"Exporting {model_key} to ONNX...")
    print(f"{'='*60}\n")
    
    # Dummy input
    dummy_input = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
    
    # Dynamic axes for batch size
    dynamic_axes = {
        'input': {0: 'batch_size'},
        'boxes': {0: 'batch_size'},
        'labels': {0: 'batch_size'},
        'scores': {0: 'batch_size'}
    }
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['boxes', 'labels', 'scores'],
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    
    print(f"✅ ONNX exported: {output_path}")
    
    # Verify
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verified")
    except ImportError:
        print(f"⚠️  onnx package not installed, skipping verification")
    except Exception as e:
        print(f"⚠️  ONNX verification failed: {e}")
    
    # Simplify (optional)
    if simplify:
        try:
            from onnxsim import simplify as onnx_simplify
            
            print(f"📦 Simplifying ONNX graph...")
            onnx_model = onnx.load(output_path)
            model_simplified, check = onnx_simplify(onnx_model)
            
            if check:
                simplified_path = output_path.replace('.onnx', '_simplified.onnx')
                onnx.save(model_simplified, simplified_path)
                print(f"✅ Simplified ONNX: {simplified_path}")
            else:
                print(f"⚠️  Simplification failed")
        except ImportError:
            print(f"⚠️  onnx-simplifier not installed, skipping simplification")
    
    # Model size
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"📊 ONNX size: {size_mb:.2f} MB")


def export_tensorrt(onnx_path, output_path, fp16=True, int8=False, calibration_data=None):
    """
    Export ONNX to TensorRT engine
    
    Args:
        onnx_path: Input .onnx file
        output_path: Output .engine file
        fp16: Enable FP16 precision
        int8: Enable INT8 quantization (requires calibration)
        calibration_data: Path to calibration images (for INT8)
    """
    print(f"\n{'='*60}")
    print(f"Exporting ONNX to TensorRT...")
    print(f"{'='*60}\n")
    
    try:
        import tensorrt as trt
    except ImportError:
        print(f"❌ TensorRT not installed. Install with:")
        print(f"   pip install nvidia-tensorrt")
        return
    
    # Logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # Builder
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    print(f"📦 Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(f"❌ Failed to parse ONNX model")
    
    print(f"✅ ONNX parsed successfully")
    
    # Config
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30  # 1GB
    
    # FP16
    if fp16 and builder.platform_has_fast_fp16:
        print(f"✅ Enabling FP16 precision")
        config.set_flag(trt.BuilderFlag.FP16)
    
    # INT8 (requires calibration)
    if int8:
        if not builder.platform_has_fast_int8:
            print(f"⚠️  Platform doesn't support INT8, falling back to FP16")
        elif calibration_data is None:
            print(f"⚠️  INT8 requires calibration data, skipping")
        else:
            print(f"✅ Enabling INT8 quantization")
            config.set_flag(trt.BuilderFlag.INT8)
            
            # TODO: Implement calibration (requires custom calibrator)
            print(f"⚠️  INT8 calibration not implemented yet")
    
    # Build engine
    print(f"🔨 Building TensorRT engine (this may take a few minutes)...")
    engine = builder.build_engine(network, config)
    
    if engine is None:
        sys.exit(f"❌ Failed to build TensorRT engine")
    
    # Serialize
    print(f"💾 Serializing engine...")
    with open(output_path, 'wb') as f:
        f.write(engine.serialize())
    
    print(f"✅ TensorRT engine saved: {output_path}")
    
    # Model size
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"📊 TensorRT engine size: {size_mb:.2f} MB")


def export_tflite(model, model_key, input_size, output_path, quantize='float32', calibration_data=None):
    """
    Export to TFLite format with optional INT8 quantization
    
    Args:
        model: PyTorch model
        model_key: Model name
        input_size: Input resolution
        output_path: Output .tflite file
        quantize: 'float32', 'float16', or 'int8'
        calibration_data: Path to calibration images (for INT8)
    """
    print(f"\n{'='*60}")
    print(f"Exporting {model_key} to TFLite ({quantize})...")
    print(f"{'='*60}\n")
    
    try:
        import tensorflow as tf
        import onnx
        from onnx_tf.backend import prepare
    except ImportError:
        print(f"❌ Required packages not installed. Install with:")
        print(f"   pip install tensorflow onnx onnx-tf")
        return
    
    # Step 1: Export to ONNX (temporary)
    temp_onnx = output_path.replace('.tflite', '_temp.onnx')
    export_onnx(model, model_key, input_size, temp_onnx, simplify=False)
    
    # Step 2: Convert ONNX to TensorFlow SavedModel
    print(f"\n📦 Converting ONNX to TensorFlow...")
    onnx_model = onnx.load(temp_onnx)
    tf_rep = prepare(onnx_model)
    
    temp_savedmodel = output_path.replace('.tflite', '_temp_savedmodel')
    tf_rep.export_graph(temp_savedmodel)
    
    print(f"✅ TensorFlow SavedModel created")
    
    # Step 3: Convert SavedModel to TFLite
    print(f"\n📦 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_saved_model(temp_savedmodel)
    
    # Quantization
    if quantize == 'float16':
        print(f"✅ Applying FP16 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    
    elif quantize == 'int8':
        print(f"✅ Applying INT8 quantization")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset (required for INT8)
        if calibration_data:
            def representative_dataset():
                calib_images = list(Path(calibration_data).glob('*.jpg'))[:100]  # Use 100 images
                for img_path in calib_images:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize((input_size, input_size))
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)
                    yield [img_array]
            
            converter.representative_dataset = representative_dataset
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.inference_input_type = tf.uint8
            converter.inference_output_type = tf.uint8
            
            print(f"✅ Using {len(list(Path(calibration_data).glob('*.jpg'))[:100])} calibration images")
        else:
            print(f"⚠️  No calibration data provided, using dynamic range quantization")
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved: {output_path}")
    
    # Cleanup
    os.remove(temp_onnx)
    import shutil
    shutil.rmtree(temp_savedmodel)
    
    # Model size
    size_mb = os.path.getsize(output_path) / (1024 ** 2)
    print(f"📊 TFLite size: {size_mb:.2f} MB")


def export_coreml(model, model_key, input_size, output_path):
    """
    Export to CoreML format (iOS/macOS)
    
    Args:
        model: PyTorch model
        model_key: Model name
        input_size: Input resolution
        output_path: Output .mlpackage directory
    """
    print(f"\n{'='*60}")
    print(f"Exporting {model_key} to CoreML...")
    print(f"{'='*60}\n")
    
    try:
        import coremltools as ct
    except ImportError:
        print(f"❌ coremltools not installed. Install with:")
        print(f"   pip install coremltools")
        return
    
    # Trace model
    print(f"📦 Tracing PyTorch model...")
    dummy_input = torch.randn(1, 3, input_size, input_size)
    traced_model = torch.jit.trace(model.cpu(), dummy_input)
    
    # Convert
    print(f"📦 Converting to CoreML...")
    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.ImageType(name="input", shape=(1, 3, input_size, input_size))],
        convert_to="mlprogram",  # Use ML Program (iOS 15+)
    )
    
    # Metadata
    mlmodel.author = "CropHealth Detection"
    mlmodel.short_description = f"{model_key} object detection model"
    mlmodel.version = "1.0"
    
    # Save
    mlmodel.save(output_path)
    
    print(f"✅ CoreML model saved: {output_path}")
    
    # Model size
    if os.path.isdir(output_path):
        import shutil
        size_mb = sum(f.stat().st_size for f in Path(output_path).rglob('*') if f.is_file()) / (1024 ** 2)
    else:
        size_mb = os.path.getsize(output_path) / (1024 ** 2)
    
    print(f"📊 CoreML size: {size_mb:.2f} MB")


def export_yolov8n(checkpoint_path, export_format, quantize='float32', calibration_data=None, output_dir=None):
    """
    Export YOLOv8n using Ultralytics built-in export
    
    Args:
        checkpoint_path: Path to best.pt
        export_format: 'onnx', 'tflite', 'coreml', 'engine' (TensorRT)
        quantize: 'float32', 'float16', 'int8'
        calibration_data: Path to calibration images
        output_dir: Output directory
    """
    print(f"\n{'='*60}")
    print(f"Exporting YOLOv8n to {export_format.upper()}...")
    print(f"{'='*60}\n")
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"❌ ultralytics not installed")
        return
    
    # Load model
    model = YOLO(checkpoint_path)
    
    # Export parameters
    export_params = {
        'format': export_format,
        'imgsz': 640,
        'half': (quantize == 'float16'),
        'int8': (quantize == 'int8'),
    }
    
    # INT8 calibration
    if quantize == 'int8' and calibration_data:
        export_params['data'] = calibration_data  # Ultralytics handles calibration
    
    # Export
    exported_path = model.export(**export_params)
    
    print(f"✅ YOLOv8n exported: {exported_path}")
    
    # Model size
    if os.path.exists(exported_path):
        size_mb = os.path.getsize(exported_path) / (1024 ** 2)
        print(f"📊 Model size: {size_mb:.2f} MB")
    
    return exported_path


def main(args):
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"🌾 CropHealth Detection - Model Export")
    print(f"Model: {args.model}")
    print(f"Format: {args.format.upper()}")
    print(f"Quantization: {args.quantize}")
    print(f"{'='*60}")
    
    # YOLOv8n uses Ultralytics export
    if args.model == 'yolov8n':
        if args.format == 'tensorrt':
            export_format = 'engine'
        else:
            export_format = args.format
        
        export_yolov8n(
            args.checkpoint,
            export_format,
            args.quantize,
            args.calibration_data,
            output_dir
        )
        return
    
    # Other models
    device = torch.device('cpu')  # Export on CPU for compatibility
    config = MODEL_CONFIGS[args.model]
    input_size = config['input_size']
    
    # Load model
    print(f"\n📦 Loading model...")
    model = build_model(args.model, args.checkpoint, device)
    print(f"✅ Model loaded")
    
    # Export based on format
    model_name = config['name']
    
    if args.format == 'onnx':
        output_path = output_dir / f"{model_name}.onnx"
        export_onnx(model, args.model, input_size, str(output_path))
        
        # Optional: Export to TensorRT from ONNX
        if args.tensorrt_from_onnx:
            engine_path = output_dir / f"{model_name}.engine"
            export_tensorrt(
                str(output_path),
                str(engine_path),
                fp16=(args.quantize == 'float16'),
                int8=(args.quantize == 'int8'),
                calibration_data=args.calibration_data
            )
    
    elif args.format == 'tensorrt':
        # First export to ONNX, then TensorRT
        onnx_path = output_dir / f"{model_name}.onnx"
        export_onnx(model, args.model, input_size, str(onnx_path))
        
        engine_path = output_dir / f"{model_name}.engine"
        export_tensorrt(
            str(onnx_path),
            str(engine_path),
            fp16=(args.quantize == 'float16'),
            int8=(args.quantize == 'int8'),
            calibration_data=args.calibration_data
        )
    
    elif args.format == 'tflite':
        output_path = output_dir / f"{model_name}_{args.quantize}.tflite"
        export_tflite(
            model,
            args.model,
            input_size,
            str(output_path),
            args.quantize,
            args.calibration_data
        )
    
    elif args.format == 'coreml':
        output_path = output_dir / f"{model_name}.mlpackage"
        export_coreml(model, args.model, input_size, str(output_path))
    
    else:
        print(f"❌ Unknown format: {args.format}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"✅ Export complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CropHealth Model Export',
                                     usage="""
CropHealth Detection - Model Export
Export models to ONNX, TensorRT, TFLite (INT8), CoreML for mobile/edge deployment

Usage:
    # Export SSD to ONNX
    python export/export_models.py --model ssd \
                                    --checkpoint runs/CropHealth_SSD/best.pt \
                                    --format onnx \
                                    --output exports/ssd

    # Export YOLOv8n to TFLite INT8
    python export/export_models.py --model yolov8n \
                                    --checkpoint runs/CropHealth_YOLOv8n/weights/best.pt \
                                    --format tflite \
                                    --quantize int8 \
                                    --calibration-data data/yolo_crop/train/images \
                                    --output exports/yolov8n

    # Export Faster R-CNN to TensorRT
    python export/export_models.py --model fasterrcnn \
                                    --checkpoint runs/CropHealth_FasterRCNN/best.pt \
                                    --format tensorrt \
                                    --output exports/fasterrcnn
""")
    parser.add_argument('--model', type=str, required=True,
                        choices=['ssd', 'yolov8n', 'efficientdet', 'fasterrcnn', 'fasterrcnn_light'],
                        help='Model architecture')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--format', type=str, required=True,
                        choices=['onnx', 'tensorrt', 'tflite', 'coreml'],
                        help='Export format')
    parser.add_argument('--quantize', type=str, default='float32',
                        choices=['float32', 'float16', 'int8'],
                        help='Quantization precision')
    parser.add_argument('--calibration-data', type=str,
                        help='Path to calibration images (for INT8 quantization)')
    parser.add_argument('--output', type=str, default='exports',
                        help='Output directory')
    parser.add_argument('--tensorrt-from-onnx', action='store_true',
                        help='Also export TensorRT engine from ONNX')
    
    args = parser.parse_args()
    
    main(args)