#!/usr/bin/env python3
"""
CropHealth Detection - Pascal VOC to YOLO Converter (Multiprocessing)
Convertit annotations Pascal VOC XML → YOLO txt + COCO JSON avec multiprocessing

Usage:
    # Conversion YOLO
    python utils/pascalvoc_to_yolo.py \
        --voc-root data/pascal_voc \
        --output data/yolo_crop \
        --format yolo \
        --workers 4

    # Conversion COCO
    python utils/pascalvoc_to_yolo.py \
        --voc-root data/pascal_voc \
        --output data/coco_crop \
        --format coco \
        --workers 4

Structure Pascal VOC attendue:
    data/pascal_voc/
    ├── train/
    │   ├── images/
    │   │   ├── img1.jpg
    │   │   └── img2.jpg
    │   └── Annotations/
    │       ├── img1.xml
    │       └── img2.xml
    ├── val/
    │   ├── images/
    │   └── Annotations/
    └── test/  (optionnel)
        ├── images/
        └── Annotations/
"""
import argparse
import json
import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, Manager
from tqdm import tqdm
from functools import partial


def parse_voc_xml(xml_path):
    """
    Parse Pascal VOC XML annotation
    
    Returns:
        dict: {'filename': str, 'size': (w, h), 'objects': [{'name': str, 'bbox': [xmin, ymin, xmax, ymax]}]}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Image info
    filename = root.find('filename').text
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Objects
    objects = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)
        
        objects.append({
            'name': name,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    return {
        'filename': filename,
        'size': (width, height),
        'objects': objects
    }


# ============= WORKERS POUR MULTIPROCESSING =============

def process_xml_to_yolo_worker(args):
    """Worker pour convertir un XML en YOLO format"""
    xml_path, voc_images, output_images, output_labels, class_to_id = args
    
    try:
        # Parse XML
        annotation = parse_voc_xml(xml_path)
        
        # Copier image
        img_name = annotation['filename']
        img_src = voc_images / img_name
        img_dst = output_images / img_name
        
        if img_src.exists():
            shutil.copy(img_src, img_dst)
        else:
            return None, f"Image not found: {img_src}"
        
        # Convertir annotations → YOLO format
        width, height = annotation['size']
        yolo_lines = []
        
        for obj in annotation['objects']:
            class_name = obj['name']
            
            if class_name not in class_to_id:
                return None, f"Unknown class '{class_name}' in {xml_path.name}"
            
            class_id = class_to_id[class_name]
            xmin, ymin, xmax, ymax = obj['bbox']
            
            # Convertir en normalized YOLO format
            x_center = ((xmin + xmax) / 2) / width
            y_center = ((ymin + ymax) / 2) / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            # Clamp to [0, 1]
            x_center = max(0, min(1, x_center))
            y_center = max(0, min(1, y_center))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
        # Sauvegarder YOLO txt
        txt_name = xml_path.stem + '.txt'
        txt_path = output_labels / txt_name
        
        with open(txt_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        return xml_path.name, None
    
    except Exception as e:
        return None, f"Error processing {xml_path.name}: {str(e)}"


def process_xml_to_coco_worker(args):
    """Worker pour extraire les données d'un XML pour COCO format"""
    xml_path, voc_images, output_images, class_to_id, img_id_start = args
    
    try:
        # Parse XML
        annotation = parse_voc_xml(xml_path)
        
        # Copier image
        img_name = annotation['filename']
        img_src = voc_images / img_name
        img_dst = output_images / img_name
        
        if img_src.exists():
            shutil.copy(img_src, img_dst)
        else:
            return None, None, f"Image not found: {img_src}"
        
        # Image metadata
        width, height = annotation['size']
        
        img_data = {
            'id': img_id_start,
            'file_name': img_name,
            'width': width,
            'height': height
        }
        
        # Annotations
        annotations = []
        for obj in annotation['objects']:
            class_name = obj['name']
            
            if class_name not in class_to_id:
                continue
            
            class_id = class_to_id[class_name]
            xmin, ymin, xmax, ymax = obj['bbox']
            
            # COCO bbox format: [x, y, width, height]
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            
            annotations.append({
                'image_id': img_id_start,
                'category_id': class_id,
                'bbox': [xmin, ymin, bbox_w, bbox_h],
                'area': bbox_w * bbox_h,
                'iscrowd': 0
            })
        
        return img_data, annotations, None
    
    except Exception as e:
        return None, None, f"Error processing {xml_path.name}: {str(e)}"


# ============= CONVERSIONS AVEC MULTIPROCESSING =============

def convert_to_yolo(voc_root, output_root, class_names, num_workers=4):
    """
    Convertit Pascal VOC → YOLO format avec multiprocessing
    
    Output structure:
        output_root/
        ├── train/
        │   ├── images/
        │   │   └── img1.jpg
        │   └── labels/
        │       └── img1.txt  # class_id x_center y_center width height (normalized)
        ├── val/
        └── data.yaml
    """
    voc_root = Path(voc_root)
    output_root = Path(output_root)
    
    # Mapping class name → class ID
    class_to_id = {name: i for i, name in enumerate(class_names)}
    
    print(f"\n{'='*60}")
    print(f"Converting Pascal VOC → YOLO")
    print(f"Input: {voc_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {class_names}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    for split in ['train', 'val', 'test']:
        voc_split_dir = voc_root / split
        
        if not voc_split_dir.exists():
            print(f"⚠️  Skipping {split} (not found)")
            continue
        
        print(f"📦 Processing {split} set...")
        
        # Dossiers output
        output_images = output_root / split / 'images'
        output_labels = output_root / split / 'labels'
        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)
        
        # Parcourir annotations XML
        voc_annotations = voc_split_dir / 'Annotations'
        voc_images = voc_split_dir / 'images'
        
        xml_files = list(voc_annotations.glob('*.xml'))
        
        # Préparer arguments pour multiprocessing
        args_list = [
            (xml_path, voc_images, output_images, output_labels, class_to_id)
            for xml_path in xml_files
        ]
        
        # Traitement parallèle
        success_count = 0
        error_count = 0
        
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_xml_to_yolo_worker, args_list),
                total=len(args_list),
                desc=f"Converting {split}"
            ))
        
        # Compter succès/erreurs
        for result, error in results:
            if error:
                error_count += 1
                if error_count <= 5:  # Afficher max 5 erreurs
                    print(f"  ⚠️  {error}")
            else:
                success_count += 1
        
        print(f"✅ {split}: {success_count}/{len(xml_files)} annotations converted")
        if error_count > 5:
            print(f"  ⚠️  ... et {error_count - 5} autres erreurs")
    
    # Créer data.yaml pour YOLOv8
    create_yolo_yaml(output_root, class_names)
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 YOLO dataset: {output_root}")


def convert_to_coco(voc_root, output_root, class_names, num_workers=4):
    """
    Convertit Pascal VOC → COCO JSON format avec multiprocessing
    
    Output structure:
        output_root/
        ├── train/
        │   ├── images/
        │   └── annotations.json
        ├── val/
        │   ├── images/
        │   └── annotations.json
        └── test/ (optionnel)
    """
    voc_root = Path(voc_root)
    output_root = Path(output_root)
    
    # Mapping class name → class ID
    class_to_id = {name: i + 1 for i, name in enumerate(class_names)}  # 1-indexed pour COCO
    
    print(f"\n{'='*60}")
    print(f"Converting Pascal VOC → COCO JSON")
    print(f"Input: {voc_root}")
    print(f"Output: {output_root}")
    print(f"Classes: {class_names}")
    print(f"Workers: {num_workers}")
    print(f"{'='*60}\n")
    
    for split in ['train', 'val', 'test']:
        voc_split_dir = voc_root / split
        
        if not voc_split_dir.exists():
            print(f"⚠️  Skipping {split} (not found)")
            continue
        
        print(f"📦 Processing {split} set...")
        
        # Dossiers output
        output_images = output_root / split / 'images'
        output_images.mkdir(parents=True, exist_ok=True)
        
        # Structure COCO
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': []
        }
        
        # Catégories
        for class_name, class_id in class_to_id.items():
            coco_data['categories'].append({
                'id': class_id,
                'name': class_name,
                'supercategory': 'crop'
            })
        
        # Parcourir annotations
        voc_annotations = voc_split_dir / 'Annotations'
        voc_images = voc_split_dir / 'images'
        
        xml_files = list(voc_annotations.glob('*.xml'))
        
        # Préparer arguments pour multiprocessing
        args_list = [
            (xml_path, voc_images, output_images, class_to_id, i + 1)
            for i, xml_path in enumerate(xml_files)
        ]
        
        # Traitement parallèle
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_xml_to_coco_worker, args_list),
                total=len(args_list),
                desc=f"Converting {split}"
            ))
        
        # Agréger les résultats
        ann_id = 1
        success_count = 0
        error_count = 0
        
        for img_data, annotations, error in results:
            if error:
                error_count += 1
                if error_count <= 5:
                    print(f"  ⚠️  {error}")
                continue
            
            if img_data:
                coco_data['images'].append(img_data)
                
                for ann in annotations:
                    ann['id'] = ann_id
                    coco_data['annotations'].append(ann)
                    ann_id += 1
                
                success_count += 1
        
        # Sauvegarder JSON
        json_path = output_root / split / 'annotations.json'
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
        
        print(f"✅ {split}: {len(coco_data['images'])} images, {len(coco_data['annotations'])} annotations")
        if error_count > 5:
            print(f"  ⚠️  ... et {error_count - 5} autres erreurs")
    
    print(f"\n✅ Conversion complete!")
    print(f"📁 COCO dataset: {output_root}")


def create_yolo_yaml(output_root, class_names):
    """Crée data.yaml pour YOLOv8"""
    yaml_content = f"""# CropHealth Detection - YOLO Dataset Config
path: {output_root.absolute()}
train: train/images
val: val/images
test: test/images  # optionnel

# Classes
nc: {len(class_names)}
names: {class_names}
"""
    os.makedirs(output_root, exist_ok=True)
    yaml_path = output_root / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"✅ Created: {yaml_path}")


def extract_class_names_from_voc(voc_root):
    """Extrait automatiquement les noms de classes depuis annotations XML"""
    voc_root = Path(voc_root)
    class_names_set = set()
    
    for split in ['train', 'val', 'test']:
        annotations_dir = voc_root / split / 'Annotations'
        
        if not annotations_dir.exists():
            continue
        
        for xml_path in annotations_dir.glob('*.xml'):
            annotation = parse_voc_xml(xml_path)
            for obj in annotation['objects']:
                class_names_set.add(obj['name'])
    
    class_names = sorted(list(class_names_set))
    return class_names


def main(args):
    voc_root = Path(args.voc_root)
    
    # Vérifier structure Pascal VOC
    print(f"\n{'='*60}")
    print(f"🔍 Checking Pascal VOC structure...")
    print(f"{'='*60}\n")
    
    for split in ['train', 'val', 'test']:
        annotations = voc_root / split / 'Annotations'
        images = voc_root / split / 'images'
        
        if annotations.exists() and images.exists():
            xml_count = len(list(annotations.glob('*.xml')))
            img_count = len(list(images.glob('*.jpg'))) + len(list(images.glob('*.png')))
            print(f"✅ {split}: {xml_count} XML, {img_count} images")
        else:
            print(f"⚠️  {split}: Not found")
    
    # Extraire classes automatiquement ou utiliser celles fournies
    if args.classes:
        class_names = args.classes
        print(f"\n📋 Using provided classes: {class_names}")
    else:
        print(f"\n📋 Extracting classes from annotations...")
        class_names = extract_class_names_from_voc(voc_root)
        print(f"✅ Found {len(class_names)} classes: {class_names}")
    
    # Convertir selon format
    if args.format == 'yolo':
        convert_to_yolo(voc_root, args.output, class_names, args.workers)
    elif args.format == 'coco':
        convert_to_coco(voc_root, args.output, class_names, args.workers)
    elif args.format == 'both':
        # Convertir les deux
        yolo_output = Path(args.output) / 'yolo'
        coco_output = Path(args.output) / 'coco'
        
        convert_to_yolo(voc_root, yolo_output, class_names, args.workers)
        convert_to_coco(voc_root, coco_output, class_names, args.workers)
    else:
        raise ValueError(f"Unknown format: {args.format}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pascal VOC to YOLO/COCO Converter (Multiprocessing)')
    parser.add_argument('--voc-root', type=str, required=True,
                        help='Pascal VOC dataset root (contains train/val/test)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--format', type=str, default='yolo',
                        choices=['yolo', 'coco', 'both'],
                        help='Output format')
    parser.add_argument('--classes', type=str, nargs='+',
                        help='Class names (auto-extracted if not provided)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of CPU workers for multiprocessing (default: 4)')
    
    args = parser.parse_args()
    main(args)