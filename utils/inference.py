import tensorflow as tf
import numpy as np

# ==============================
# 1. Fonction d'évaluation depuis un checkpoint
# ==============================
def evaluate_model_from_checkpoint(model_class, checkpoint_path, test_dataset, iou_threshold=0.5):
    """
    Charge un modèle depuis un checkpoint et évalue ses performances.
    
    Args:
        model_class: classe du modèle (ex: YOLOv8n, SSD, FasterRCNN).
        checkpoint_path: chemin vers le checkpoint sauvegardé.
        test_dataset: dataset de test (images + annotations).
        iou_threshold: seuil IoU pour calculer le mAP@50.
    
    Returns:
        dict avec précision, rappel, F1-score, mAP@50.
    """
    # Charger le modèle
    model = model_class()
    model.load_weights(checkpoint_path)
    
    # Variables pour accumuler les résultats
    all_precisions, all_recalls, all_f1, all_maps = [], [], [], []
    
    for images, labels in test_dataset:
        preds = model.predict(images)
        
        # Ici tu peux brancher ta fonction de calcul mAP/precision/recall
        # Exemple simplifié :
        precision = np.random.uniform(0.7, 0.95)  # à remplacer par ton calcul réel
        recall = np.random.uniform(0.7, 0.95)
        f1 = 2 * (precision * recall) / (precision + recall)
        map50 = np.random.uniform(0.7, 0.95)
        
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1.append(f1)
        all_maps.append(map50)
    
    return {
        "precision": np.mean(all_precisions),
        "recall": np.mean(all_recalls),
        "f1_score": np.mean(all_f1),
        "mAP@50": np.mean(all_maps)
    }

# ==============================
# 2. Fonction d’export en TensorFlow Lite int8
# ==============================
def export_to_tflite_int8(model, representative_data_gen, output_path="model_int8.tflite"):
    """
    Exporte un modèle TensorFlow en TensorFlow Lite avec quantization int8.
    
    Args:
        model: modèle TensorFlow entraîné.
        representative_data_gen: générateur de données représentatives pour calibrer la quantization.
        output_path: chemin du fichier TFLite exporté.
    
    Returns:
        Sauvegarde un fichier .tflite optimisé.
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"✅ Modèle exporté en TensorFlow Lite int8 : {output_path}")

if __name__ == "__main__":
    # Exemple d'utilisation (à adapter selon ton contexte)
    
    # 1. Évaluation du modèle
    # from some_model_library import YOLOv8n  # Remplace par ta classe de modèle
    # test_dataset = ...  # Prépare ton dataset de test ici
    # results = evaluate_model_from_checkpoint(YOLOv8n, "path/to/checkpoint", test_dataset)
    # print("Résultats d'évaluation :", results)
    
    # 2. Export en TFLite int8
    # model = ...  # Charge ou crée ton modèle TensorFlow ici
    # def representative_data_gen():
    #     for _ in range(100):
    #         data = np.random.rand(1, 224, 224, 3).astype(np.float32)  # Exemple de données
    #         yield [data]
    # export_to_tflite_int8(model, representative_data_gen, "model_int8.tflite")
    pass    