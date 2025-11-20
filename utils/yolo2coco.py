#!/usr/bin/env python3
"""
CropHealth Detection - YOLO to COCO Converter
Convertit annotations YOLO txt → COCO JSON pour EfficientDet-D0

Usage:
    python utils/yolo2coco.py --yolo-root data/yolo_crop --output data/coco_crop

Structure YOLO attendue:
    yolo_root/
        train/
            images/
            labels/
        val/
            images/
            labels/

Structure COCO générée:
    output/
        train/
            images/
            annotations.json
        val/
            images/
            annotations.json
"""
import argparse
import json
import shutil
from pathlib import Path
from PIL import Image


def convert_yolo_to_coco(yolo_root, output_root, split='train', class_names=None):
    """
    Convertit YOLO → COCO pour un split (train/val)
    
    Args:
        yolo_root: racine dataset YOLO
        output_root: racine output COCO
        split: 'train' ou 'val'
        class_names: liste des noms de classes
    """
    yolo_img_dir = Path(yolo_root) / split / 'images'
    yolo_lbl_dir = Path(yolo_root) / split / 'labels'
    
    coco_img_dir = Path(output_root) / split / 'images'
    coco_json = Path(output_root) / split / 'annotations.json'
    
    # Créer dossiers output
    coco_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialiser structure COCO
    coco_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # Catégories (1-indexed pour COCO)
    if class_names is None:
        class_names = [f'class_{i}' for i in range(9)]
    
    for i, name in enumerate(class_names):
        coco_data['categories'].append({
            'id': i + 1,
            'name': name,
            'supercategory': 'crop'
        })
    
    # Parcourir images
    img_id = 1
    ann_id = 1
    
    for img_path in sorted(yolo_img_dir.glob('*.jpg')):
        # Copier image
        shutil.copy(img_path, coco_img_dir / img_path.name)
        
        # Métadonnées image
        img = Image.open(img_path)
        w, h = img.size
        
        coco_data['images'].append({
            'id': img_id,
            'file_name': img_path.name,
            'width': w,
            'height': h
        })
        
        # Lire annotations YOLO
        lbl_path = yolo_lbl_dir / (img_path.stem + '.txt')
        
        if lbl_path.exists():
            for line in open(lbl_path):
                cls, xc, yc, ww, hh = map(float, line.strip().split())
                
                # YOLO normalized → COCO absolute xywh
                x = (xc - ww / 2) * w
                y = (yc - hh / 2) * h
                box_w = ww * w
                box_h = hh * h
                
                coco_data['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': int(cls) + 1,  # 1-indexed
                    'bbox': [x, y, box_w, box_h],
                    'area': box_w * box_h,
                    'iscrowd': 0
                })
                ann_id += 1
        
        img_id += 1
    
    # Sauvegarder JSON
    with open(coco_json, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
    print(f"   Saved: {coco_json}")


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO to COCO format')
    parser.add_argument('--yolo-root', type=str, required=True,
                        help='YOLO dataset root (contains train/val)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output COCO dataset root')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='Class names (optional, defaults to class_0, class_1, ...)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Converting YOLO → COCO")
    print(f"Input: {args.yolo_root}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")
    
    # Convertir train + val
    for split in ['train', 'val']:
        convert_yolo_to_coco(args.yolo_root, args.output, split, args.classes)
    
    print(f"\n✅ Conversion complete!")
    print(f"Use with: python train.py --model efficientdet --data {args.output}\n")


if __name__ == '__main__':
    main()