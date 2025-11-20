"""
Wrapper pour YOLOv8n - Compatibilité avec evaluate_models.py
Convertit outputs Ultralytics en format PyTorch standard
"""
import torch
from ultralytics import YOLO


class YOLOv8Wrapper:
    """Wrapper pour utiliser YOLOv8n comme modèles PyTorch"""
    
    def __init__(self, checkpoint_path):
        self.model = YOLO(checkpoint_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def to(self, device):
        self.device = str(device)
        return self
    
    def eval(self):
        return self
    
    def __call__(self, images):
        """
        Forward pass compatible avec evaluate_models.py
        
        Args:
            images: list de tensors [C, H, W]
        
        Returns:
            list de dicts {'boxes': Tensor, 'labels': Tensor, 'scores': Tensor}
        """
        predictions = []
        
        for img in images:
            # Ultralytics attend format [H, W, C] ou path
            img_np = img.cpu().permute(1, 2, 0).numpy()
            
            # Prédiction
            results = self.model(img_np, verbose=False)
            
            # Extraire boxes, labels, scores
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy  # format xyxy
                labels = results[0].boxes.cls.long() + 1  # +1 pour background=0
                scores = results[0].boxes.conf
                
                predictions.append({
                    'boxes': boxes.cpu(),
                    'labels': labels.cpu(),
                    'scores': scores.cpu()
                })
            else:
                # Pas de détection
                predictions.append({
                    'boxes': torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.long),
                    'scores': torch.zeros((0,))
                })
        
        return predictions