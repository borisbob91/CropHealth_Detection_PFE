"""
CropHealth Detection - Pascal VOC Dataset
Charge directement les annotations Pascal VOC XML (sans conversion)
Utilisé par: SSD, Faster R-CNN, Faster R-CNN light
"""
import warnings
import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

from datasets.transforms import validate_bbox_format


class PascalVOCDataset(Dataset):
    """
    Dataset Pascal VOC format (XML) → PyTorch
    Applique Albumentations transforms
    
    Structure attendue:
        data_root/
        ├── images/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── Annotations/
            ├── img1.xml
            └── img2.xml
    """
    def __init__(self, img_root, ann_root, class_names, transforms=None):
        """
        Args:
            img_root: chemin vers images/
            ann_root: chemin vers Annotations/
            class_names: liste des noms de classes (ex: ['aphid', 'bollworm', ...])
            transforms: pipeline Albumentations (get_albu_transform)
        """
        self.img_root = Path(img_root)
        self.ann_root = Path(ann_root)
        self.class_names = class_names
        self.class_to_idx = {name: i + 1 for i, name in enumerate(class_names)}  # background = 0
        self.transforms = transforms
        
        # Lister annotations XML
        self.annotations = sorted(list(self.ann_root.glob('*.xml')))
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # Charger annotation XML
        xml_path = self.annotations[idx]
        annotation = self._parse_xml(xml_path)
        
        # Charger image
        img_name = annotation['filename']
        img_path = self.img_root / img_name
        img = np.array(Image.open(img_path).convert('RGB'))
        
        # Extraire boxes et labels
        boxes = []
        labels = []
        
        for obj in annotation['objects']:
            class_name = obj['name']
            
            # Ignorer classes inconnues
            if class_name not in self.class_to_idx:
                continue
            
            boxes.append(obj['bbox'])
            labels.append(self.class_to_idx[class_name])
        filename = annotation['filename']
        img_shape = img.shape[1:]  # (H, W)
        
        validated_boxes = validate_bbox_format(boxes, img_shape, filename)
        
        if validated_boxes is None:
            # ❌ Boîtes invalides - retourner image vide
            warnings.warn(f"Image {filename} ignorée car boîtes non normalisées")
            return self._get_empty_sample(img_shape)
        # Gestion images sans annotations valides
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]
            labels = [0]
        
        # Appliquer Albumentations
        if self.transforms:
            transformed = self.transforms(
                image=img,
                bboxes=boxes,
                class_labels=labels
            )
            img = transformed['image']
            boxes = transformed['bboxes']
            labels = transformed['class_labels']
        
        # Conversion en tenseurs PyTorch
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
        }
        return img, target
    
    def _parse_xml(self, xml_path):
        """Parse Pascal VOC XML annotation"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Image info
        filename = root.find('filename').text
        
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
            'objects': objects
        }
    
    def get_class_names(self):
        """Retourne les noms de classes"""
        return self.class_names
    
    def _get_empty_sample(self, img_shape):
        """Retourne un échantillon vide pour les cas invalides"""
        h, w = img_shape
        
        # Boîte factice
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.zeros(0, dtype=torch.int64)
        
        target = {'boxes': boxes, 'labels': labels}
        
        # Image vide (noire)
        img = torch.zeros((3, h, w), dtype=torch.float32)
        
        return img, target