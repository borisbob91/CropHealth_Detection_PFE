"""
CropHealth Detection - Model Configurations
Centralise tous les hyperparamètres selon Tableau 6 du rapport
"""


CLASS_NAMES = [
    'A. flava',
    'Adulte derogata',
    'B. tabaci',
    'Coccinelle',
    'Degat A. flava',
    'Degat Jassides',
    'Dysdercus spp',
    'Earias spp',
    'Effet phyto',
    'Fourmie',
    'G. spodoctera',
    'H. amirgera',
    'Jasside',
    'Larve coccinelle',
    'Larve syrphe',
    'P. gossypiella',
    'Puceron',
    'S. derogata',
    'S. frugiperda',
    'Scarabees',
    'Syrphe',
    'punaisse_01',
    'punaisse_02',
    'punaisse_03',
    'punaisse_04'
]


NUM_CLASSES = len(CLASS_NAMES) + 1  # +1 pour background

MODEL_CONFIGS = {
    'ssd': {
        'name': 'CropHealth_SSD',
        'backbone': 'MobileNetV3',
        'input_size': 320,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.005,
        'optimizer': 'SGD',
        'weight_decay': 1e-4,
        'scheduler': 'cosine', # 
        'dataset_format': 'yolo',  # 'yolo', 'coco', ou 'pascalvoc'
        'weights': 'SSDLite320_MobileNet_V3_Large_Weights.COCO_V1',
    },
    'yolov8n': {
        'name': 'CropHealth_YOLOv8n',
        'backbone': 'CSP-Darknet',
        'input_size': 640,
        'epochs': 2,
        'batch_size': 32,
        'lr': 0.01,
        'optimizer': 'SGD',
        'weight_decay': 5e-4,
        'scheduler': 'cosine',
        'dataset_format': 'yolo',  # Direct via ultralytics
        'weights': 'yolov8n.pt',
    },
     'yolov11n': {  # NOUVEAU !
        'name': 'CropHealth_YOLOv11n',
        'backbone': 'CSP-Darknet',
        'input_size': 640,
        'epochs': 50,
        'batch_size': 32,
        'lr': 0.01,
        'optimizer': 'SGD',
        'weight_decay': 5e-4,
        'scheduler': 'cosine',
        'dataset_format': 'yolo',
        'weights': 'yolo11n.pt',
    },
    'efficientdet': {
        'name': 'CropHealth_EfficientDet',
        'backbone': 'EfficientNet-B0',
        'input_size': 512,
        'epochs': 50,
        'batch_size': 8,
        'lr': 0.001,
        'optimizer': 'AdamW',
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'dataset_format': 'coco',  # COCO JSON
        'weights': 'tf_efficientdet_d0',  # via timm
    },
    'fasterrcnn': {
        'name': 'CropHealth_FasterRCNN',
        'backbone': 'ResNet50+FPN',
        'input_size': 800,
        'epochs': 12,
        'batch_size': 4,
        'lr': 0.005,
        'optimizer': 'SGD',
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'dataset_format': 'yolo',  # YOLO txt
        'weights': 'FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1',
    },
    'fasterrcnn_light': {
        'name': 'CropHealth_FasterRCNN_light',
        'backbone': 'MobileNetV3+FPN',
        'input_size': 320,
        'epochs': 20,
        'batch_size': 16,
        'lr': 0.005,
        'optimizer': 'SGD',
        'weight_decay': 1e-4,
        'scheduler': 'cosine',
        'dataset_format': 'yolo',  # YOLO txt
        'weights': 'FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1',
    },
}
