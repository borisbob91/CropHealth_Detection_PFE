"""
CropHealth Detection - YOLO Dataset
Charge les annotations YOLO txt et applique Albumentations
Utilisé par: SSD, Faster R-CNN, Faster R-CNN light
"""
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class YoloDataset(Dataset):
    """
    Dataset YOLO format (txt) → Pascal VOC boxes + Albumentations
    
    Structure attendue:
        data_root/
            images/
                img1.jpg
                img2.jpg
            labels/
                img1.txt  # cls xc yc w h (normalized)
                img2.txt
    """
    def __init__(self, img_root, lbl_root, transforms=None):
        """
        Args:
            img_root: chemin vers dossier images/
            lbl_root: chemin vers dossier labels/
            transforms: pipeline Albumentations (get_albu_transform)
        """
        self.imgs = sorted(Path(img_root).glob('*.jpg'))
        self.lbl_root = Path(lbl_root)
        self.transforms = transforms

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # Charger image
        img = np.array(Image.open(self.imgs[idx]).convert('RGB'))
        h0, w0 = img.shape[:2]
        
        # Charger annotations YOLO
        lbl_path = self.lbl_root / (self.imgs[idx].stem + '.txt')
        boxes, labels = [], []
        
        if lbl_path.exists():
            for line in open(lbl_path):
                cls, xc, yc, ww, hh = map(float, line.strip().split())
                # YOLO normalized → Pascal VOC absolute
                xmin = (xc - ww / 2) * w0
                ymin = (yc - hh / 2) * h0
                xmax = (xc + ww / 2) * w0
                ymax = (yc + hh / 2) * h0
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(int(cls) + 1)  # background = 0
        
        # Gestion images sans annotations
        if len(boxes) == 0:
            boxes = [[0, 0, 1, 1]]  # dummy box pour Albumentations
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