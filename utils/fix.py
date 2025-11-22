import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Tuple

# =================================================================
# 1. LOGIQUE DE TRAITEMENT PARALLÈLE (Le cœur de la correction)
# =================================================================

def process_xml_file_worker(xml_path: str, image_map: Dict[str, str]) -> int:
    """
    Tâche exécutée par chaque cœur : Corrige le <filename> pour un seul XML.
    Retourne 1 si corrigé, 0 sinon.
    """
    
    # 1. Extraire le nom de base du XML (ex: 000f247f-IMG_R_HA_109_zoom)
    base_name = os.path.splitext(os.path.basename(xml_path))[0]
    
    if base_name not in image_map:
        # L'image correspondante n'a pas été trouvée, on ignore.
        return 0

    # 2. Récupérer le vrai nom de l'image (ex: 000f247f-IMG_R_HA_109_zoom.jpg)
    vrai_nom_image = image_map[base_name]
    
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        filename_tag = root.find('filename')
        
        # S'assurer que la balise <filename> existe
        if filename_tag is None:
            filename_tag = ET.SubElement(root, 'filename')
        
        # 3. Correction et écriture
        if filename_tag.text != vrai_nom_image:
            filename_tag.text = vrai_nom_image
            tree.write(xml_path)
            # print(f"  [OK] Corrigé : {os.path.basename(xml_path)} → {vrai_nom_image}") # Désactivé pour la vitesse
            return 1 # Fichier corrigé
            
    except Exception:
        # Loguer l'erreur si nécessaire, mais retourner 0 pour continuer
        # print(f"  [ERREUR] Impossible de traiter {os.path.basename(xml_path)}")
        pass
        
    return 0 # Pas de correction effectuée

# =================================================================
# 2. FONCTION PRINCIPALE (Orchestration Multiprocessus)
# =================================================================

def corriger_filename_dans_xml_multiprocess(root_dataset_path: str):
    """
    Collecte les tâches et utilise un ProcessPoolExecutor pour les exécuter en parallèle.
    """
    splits = ['train', 'val', 'test', 'loss'] # J'ajoute 'loss' au cas où
    max_workers = os.cpu_count() or 4
    all_xml_tasks = []
    
    print(f"Démarrage de la correction XML ultra-rapide avec {max_workers} cœurs.")
    
    for split in splits:
        ann_dir = os.path.join(root_dataset_path, split, 'Annotations')
        img_dir = os.path.join(root_dataset_path, split, 'images')

        if not os.path.isdir(ann_dir) or not os.path.isdir(img_dir):
            continue
            
        print(f"→ Préparation des tâches pour le split : {split}...")

        # 1. Optimisation : Créer une MAP des images avant de lancer les processus
        # Cela évite les appels lents et redondants à os.listdir() dans les workers.
        image_map = {}
        for f in os.listdir(img_dir):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                base_name = os.path.splitext(f)[0]
                image_map[base_name] = f
        
        # 2. Collecter les chemins XML et les maps correspondantes
        for xml_file in os.listdir(ann_dir):
            if xml_file.lower().endswith('.xml'):
                xml_path = os.path.join(ann_dir, xml_file)
                # On ajoute la tâche: (chemin XML, map d'images du split)
                all_xml_tasks.append((xml_path, image_map))
    
    if not all_xml_tasks:
        print("Aucun fichier XML à traiter trouvé.")
        return

    total_corriges = 0
    total_fichiers = len(all_xml_tasks)
    
    print(f"\nCorrection de {total_fichiers} fichiers XML en parallèle...")

    # 3. Exécution parallèle
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Soumettre toutes les tâches
        futures = [executor.submit(process_xml_file_worker, xml_path, image_map) 
                   for xml_path, image_map in all_xml_tasks]

        # Récupérer les résultats au fur et à mesure (as_completed)
        for i, future in enumerate(as_completed(futures)):
            total_corriges += future.result()
            # Afficher la progression
            if (i + 1) % 500 == 0 or i + 1 == total_fichiers:
                print(f"  Progression : {i + 1}/{total_fichiers} traités... ({total_corriges} corrigés)")

    print("\n" + "="*60)
    print(f"CORRECTION TERMINÉE ! {total_corriges} fichiers XML mis à jour sur {total_fichiers} vérifiés.")
    print("La vitesse d'exécution devrait être nettement supérieure.")
    print("="*60)

# ======================== À LANCER ========================
if __name__ == "__main__":
    # METS TON DOSSIER RACINE ICI (celui avec train/val/test)
    # Assurez-vous que le chemin est correct pour votre environnement (local ou Google Drive/Colab)
    root = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc_ac"
    
    # ⚠️ Vérifiez si ce chemin est correct avant de lancer !
    if not os.path.exists(root):
        print("ERREUR : Le chemin racine du dataset n'existe pas. Veuillez le corriger.")
    else:
        corriger_filename_dans_xml_multiprocess(root)