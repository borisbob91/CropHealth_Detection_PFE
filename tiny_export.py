from ultralytics import YOLO
import torch

# ==================== ULTRALYTICS MODULES ====================
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv, Concat
from ultralytics.nn.modules.block import C2f, Bottleneck, SPPF
from ultralytics.nn.modules.head import Detect

# ==================== PYTORCH BASE MODULES ====================
from torch.nn import Sequential, ModuleList
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.activation import SiLU, Hardswish
from torch.nn.modules.pooling import MaxPool2d, AdaptiveAvgPool2d
from torch.nn.modules.upsampling import Upsample
from torch.nn.modules.linear import Linear
from ultralytics.nn.modules.block import DFL
# ==================== LISTE COMPLÈTE ====================
torch.serialization.add_safe_globals([
    # --- Ultralytics Custom ---
    DetectionModel,
    Conv,
    Concat,
    C2f,
    Bottleneck,
    SPPF,
    Detect,
    
    # --- PyTorch Core ---
    Conv2d,
    BatchNorm2d,
    SiLU,
    Hardswish,  # Optionnel mais recommandé
    MaxPool2d,
    AdaptiveAvgPool2d,  # Parfois utilisé dans SPPF
    Upsample,
    Linear,
    Sequential,
    ModuleList,
    DFL
])

print("🔄 Conversion en cours...")

try:
    model = YOLO('best.pt')
    model.export(format='tflite', imgsz=640, int8=False)
    print("✅ Conversion réussie !")
    
except Exception as e:
    print("❌ Erreur:", e)