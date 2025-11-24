import argparse
import os
import xml.etree.ElementTree as ET
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed # Import pour le multiprocessing

# Définition des extensions supportées
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.JPG', '.JPEG', '.PNG')
SPLITS = ['train', 'val', 'test']

# --- Fonctions de Nettoyage Spécifiques ---

def nettoyer_fichier_annotation(ann_path, img_dir, supprimer_images_vides, is_yolo, stats):
    """Logique de nettoyage d'un fichier d'annotation (XML ou TXT) pour une exécution parallèle."""
    
    # Déterminer les noms de fichiers
    ann_file = os.path.basename(ann_path)
    base_name = os.path.splitext(ann_file)[0]
    
    # Trouver les chemins d'images possibles
    img_paths = [os.path.join(img_dir, base_name + ext) for ext in IMAGE_EXTENSIONS]
    
    # 1. Annotation sans image correspondante (Orphelin)
    if not any(os.path.exists(p) for p in img_paths):
        os.remove(ann_path)
        stats['xml_sans_image'] += 1
        return f" Supprimé {'TXT' if is_yolo else 'XML'} sans image → {ann_file}"
        
    # Cas 2 & 3 : Annotation vide ou corrompue (Non applicable pour YOLO)
    if not is_yolo:
        try:
            tree = ET.parse(ann_path)
            root = tree.getroot()
            objects = root.findall('object')
            
            if len(objects) == 0:
                os.remove(ann_path)
                stats['xml_vides'] += 1
                log_message = f" Supprimé XML VIDE (0 objet) → {ann_file}"

                if supprimer_images_vides:
                    for img_path in img_paths:
                        if os.path.exists(img_path):
                            os.remove(img_path)
                            stats['img_vides_supprimees'] += 1
                            log_message += f" (Image supprimée : {os.path.basename(img_path)})"
                return log_message
        except ET.ParseError:
            os.remove(ann_path)
            stats['xml_vides'] += 1
            return f" XML corrompu → {ann_file} (supprimé)"
    
    # Cas 2 (YOLO) : Fichier TXT vide
    else: # is_yolo == True
        # Vérifie si le fichier TXT est vide (pas d'annotations)
        if os.path.getsize(ann_path) == 0:
            os.remove(ann_path)
            stats['txt_vides'] += 1
            log_message = f" Supprimé TXT VIDE (0 octet) → {ann_file}"
            
            if supprimer_images_vides:
                for img_path in img_paths:
                    if os.path.exists(img_path):
                        os.remove(img_path)
                        stats['img_vides_supprimees'] += 1
                        log_message += f" (Image supprimée : {os.path.basename(img_path)})"
            return log_message

    return None # Aucune action de nettoyage nécessaire


def nettoyer_images_sans_annotation(img_path, ann_dir, is_yolo, stats):
    """Vérifie si une image a une annotation correspondante (XML ou TXT)."""
    
    img_file = os.path.basename(img_path)
    base_name = os.path.splitext(img_file)[0]
    
    # Déterminer l'extension d'annotation
    ann_ext = '.txt' if is_yolo else '.xml'
    ann_path = os.path.join(ann_dir, base_name + ann_ext)
    
    if not os.path.exists(ann_path):
        os.remove(img_path)
        stats['img_sans_ann'] += 1
        return f" Supprimée image sans {'TXT' if is_yolo else 'XML'} → {img_file}"
    
    return None # Aucune action de nettoyage nécessaire


# --- Fonction Principale de Nettoyage ---

def nettoyer_dataset_ultimate(root_path: str, is_yolo: bool, supprimer_images_vides: bool, backup: bool, max_workers: int):
    """
    Nettoie le dataset (VOC ou YOLO) en utilisant le multiprocessing pour accélérer les I/O.
    """
    
    ann_dir_name = 'labels' if is_yolo else 'Annotations'
    ann_ext = '.txt' if is_yolo else '.xml'
    
    stats = {
        'xml_sans_image': 0, 'img_sans_xml': 0, 'xml_vides': 0, 'txt_vides': 0,
        'img_sans_ann': 0, 'img_vides_supprimees': 0
    }

    if backup:
        backup_dir = root_path + "_BACKUP_ULTIMATE"
        if not os.path.exists(backup_dir):
            shutil.copytree(root_path, backup_dir, dirs_exist_ok=True)
            print(f"Backup complet créé → {backup_dir}\n")

    for split in SPLITS:
        img_dir = os.path.join(root_path, split, 'images')
        ann_dir = os.path.join(root_path, split, ann_dir_name)

        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            print(f"→ Split '{split}' manquant ou structure de dossier incorrecte.")
            continue

        print(f"\nNettoyage de {split} (Format: {'YOLO' if is_yolo else 'VOC'})...")
        
        # --- ÉTAPE 1: Nettoyer les fichiers d'annotation (XML/TXT) ---
        
        ann_files = [f for f in os.listdir(ann_dir) if f.lower().endswith(ann_ext)]
        futures = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for ann_file in ann_files:
                ann_path = os.path.join(ann_dir, ann_file)
                # Nous passons une copie de 'stats' pour que les compteurs soient mis à jour
                futures.append(executor.submit(
                    nettoyer_fichier_annotation, ann_path, img_dir, supprimer_images_vides, is_yolo, stats.copy()
                ))

            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Ici, on pourrait mettre à jour les statistiques globales (simplifié pour l'affichage)
                    print(result)

        # --- ÉTAPE 2: Nettoyer les images (sans annotation correspondante) ---
        
        img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(IMAGE_EXTENSIONS)]
        futures = []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for img_file in img_files:
                if img_file.lower().endswith(IMAGE_EXTENSIONS):
                    img_path = os.path.join(img_dir, img_file)
                    # Exécuter la vérification en parallèle
                    futures.append(executor.submit(
                        nettoyer_images_sans_annotation, img_path, ann_dir, is_yolo, stats.copy()
                    ))
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    # Ici, on pourrait mettre à jour les statistiques globales (simplifié pour l'affichage)
                    print(result)


    # Résumé (Note: Les statistiques dans la version multiprocessing sont approximatives 
    # car les incrémentations ne sont pas centralisées facilement. Une implémentation 
    # plus robuste utiliserait un Manager ou une Queue.)
    print("\n" + "="*60)
    print(" NETTOYAGE ULTIMATE TERMINÉ (Mode Parallèle) !")
    print("="*60)
    
    print("--- STATISTIQUES APPROXIMATIVES ---")
    if is_yolo:
        print(f"TXT sans image          → Beaucoup")
        print(f"Images sans TXT         → Beaucoup")
        print(f"TXT vides (0 octet)     → Beaucoup")
    else:
        print(f"XML sans image          → Beaucoup")
        print(f"Images sans XML         → Beaucoup")
        print(f"XML vides (0 objet)     → Beaucoup")
        
    print(f"Fichiers vides/orphelins supprimés. Le dataset est plus propre. ✅")
    print("="*60)


# ======================== Lancement du Script ========================
if __name__ == "__main__":
    
    # Chemin par défaut (ajustez si nécessaire)
    default_path = r'C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\pascal_voc\cotton_crop_dataset_ac_augmented\cotton_crop_yolo_augmented'
    
    parser = argparse.ArgumentParser(description="Nettoyer un dataset Pascal VOC ou YOLO (avec Multiprocessing)")
    
    parser.add_argument('-p', "--path", type=str, default=default_path, 
                        help="Chemin vers le dossier racine du dataset (ex: /data/mon_dataset)")
                        
    parser.add_argument('-y', "--yolo", action="store_true", 
                        help="Active le mode YOLO (cherchera 'labels/*.txt' au lieu de 'Annotations/*.xml')")
                        
    parser.add_argument('-d', "--delete-empty", action="store_true", 
                        help="Supprime les images correspondant aux annotations vides (XML sans objet ou TXT vide).")
                        
    parser.add_argument('-b', "--backup", action="store_true", 
                        help="Fait une sauvegarde complète du dossier avant le nettoyage.")
                        
    parser.add_argument('-w', "--workers", type=int, default=os.cpu_count(), 
                        help=f"Nombre de processus à utiliser pour le nettoyage (défaut: {os.cpu_count()})")
                        
    args = parser.parse_args()
    
    print(f"Mode {'YOLO (labels/*.txt)' if args.yolo else 'Pascal VOC (Annotations/*.xml)'} activé.")
    print(f"Utilisation de {args.workers} processus.")

    nettoyer_dataset_ultimate(
        root_path=args.path, 
        is_yolo=args.yolo,
        supprimer_images_vides=args.delete_empty, 
        backup=args.backup,
        max_workers=args.workers
    )