"""
CropHealth Detection - Base Trainer
Boucle d'entraînement/validation unifiée pour SSD, EfficientDet, Faster R-CNN
"""
from typing import Union
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class BaseTrainer:
    """
    Trainer unifié pour tous les modèles (sauf YOLOv8n qui utilise ultralytics)
    """
    def __init__(self, model: Module, train_loader: DataLoader, val_loader: DataLoader, optimizer: Optimizer, scheduler: _LRScheduler, 
                 device:Union[str, int], save_dir:str, model_name:str):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.writer = SummaryWriter(save_dir)
        self.best_map = 0.0
    
    def train_one_epoch(self, epoch):
        """Entraînement sur une epoch"""
        self.model.train()
        total_loss = 0.0
        
        for i, (imgs, targets) in enumerate(self.train_loader):
            imgs = list(img.to(self.device) for img in imgs)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
            # Forward
            loss_dict = self.model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            # Backward
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            total_loss += losses.item()
            
            # Log toutes les 50 iterations
            if i % 50 == 0:
                step = epoch * len(self.train_loader) + i
                self.writer.add_scalar('train/loss', losses.item(), step)
                print(f"Epoch {epoch} [{i}/{len(self.train_loader)}] Loss: {losses.item():.4f}")
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.inference_mode()
    def evaluate(self, epoch):
        """Validation avec mAP@50"""
        self.model.eval()
        metric = MeanAveragePrecision(iou_type='bbox')
        
        for imgs, targets in self.val_loader:
            imgs = list(img.to(self.device) for img in imgs)
            
            # Inference
            preds = self.model(imgs)
            
            # Préparer pour torchmetrics (CPU)
            preds_cpu = [{k: v.cpu() for k, v in p.items()} for p in preds]
            targets_cpu = [{k: v.cpu() for k, v in t.items()} for t in targets]
            
            metric.update(preds_cpu, targets_cpu)
        
        # Calculer mAP
        results = metric.compute()
        map50 = results['map_50'].item()
        map_all = results['map'].item()
        
        # Log
        self.writer.add_scalar('val/mAP50', map50, epoch)
        self.writer.add_scalar('val/mAP', map_all, epoch)
        
        return map50, map_all
    
    def save_checkpoint(self, epoch, map50):
        """Sauvegarde le meilleur modèle"""
        if map50 > self.best_map:
            self.best_map = map50
            checkpoint = {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': epoch,
                'map50': map50,
            }
            save_path = f"{self.save_dir}/{self.model_name}_best.pt"
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model: mAP@50={map50:.4f}")
    
    def train(self, epochs):
        """Boucle d'entraînement complète"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Epochs: {epochs} | Device: {self.device}")
        print(f"Save dir: {self.save_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            avg_loss = self.train_one_epoch(epoch)
            
            # Validation
            map50, map_all = self.evaluate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Save best
            self.save_checkpoint(epoch, map50)
            
            # Log epoch
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f} | mAP@50: {map50:.4f} | mAP: {map_all:.4f}\n")
        
        self.writer.close()
        print(f"\n✅ Training complete! Best mAP@50: {self.best_map:.4f}")
        print(f"Model saved: {self.save_dir}/best.pt\n")