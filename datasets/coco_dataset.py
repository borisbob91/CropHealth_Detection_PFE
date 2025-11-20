"""
CropHealth Detection - COCO Dataset
Charge les annotations COCO JSON et applique Albumentations
Utilisé par: EfficientDet-D0
"""
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class CocoDataset(Dataset):
    """
    Dataset COCO JSON format → Pascal VOC boxes + Albumentations
    
    Structure attendue:
        data_root/
            images/
                img1.jpg
            annotations.json  # COCO format
    """
    def __init__(self, coco_json, img_dir, transforms=None):
        """
        Args:
            coco_json: chemin vers annotations.json
            img_dir: chemin vers dossier images/
            transforms: pipeline Albumentations (get_albu_transform)
        """
        with open(coco_json) as f:
            self.coco = json.load(f)
        
        # Index images par ID
        self.imgs = {img['id']: img for img in self.coco['images']}
        
        # Grouper annotations par image_id
        self.anns = {img_id: [] for img_id in self.imgs}
        for ann in self.coco['annotations']:
            self.anns[ann['image_id']].append(ann)
        
        self.img_dir = Path(img_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Récupérer métadonnées image
        img_id = list(self.imgs.keys())[idx]
        meta = self.imgs[img_id]
        
        # Charger image
        img = np.array(Image.open(self.img_dir / meta['file_name']).convert('RGB'))
        
        # Charger annotations COCO
        anns = self.anns[img_id]
        boxes, labels = [], []
        
        for ann in anns:
            # COCO xywh → Pascal VOC xyxy
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
        
        # Gestion images sans annotations
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