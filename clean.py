import os
import re
import shutil
from typing import List, Set, Tuple
from concurrent.futures import ProcessPoolExecutor

# =================================================================
# 1. CONFIGURATION
# =================================================================

# CLASSES À SUPPRIMER (Cibles basées sur le nom de fichier)
CLASSES_TO_DELETE: Set[str] = {
    'IMG_R_SY',
    'IMG_P_FN',
    'IMG_E_AF',
    'IMG_R_PUO',
}


SUBDIRECTORIES: List[str] = ['train', 'val', 'test'] 

# NOUVEAU DOSSIER DE SORTIE (Créé à la racine du script)
CLEAN_ROOT_DIR: str = r'data_cleanned'

# Regex pour détecter la classe dans le nom de fichier (fourni par l'utilisateur)
PATTERN_REGEX = re.compile(r'(?:^|-)([A-Z]+_[A-Z]+_[A-Z]+)(?:_\d+)?')
IMAGE_EXTENSIONS: Tuple[str, ...] = ('.jpg', '.jpeg', '.png')


# =================================================================
# 2. FONCTIONS DE PRÉPARATION ET DE TRAITEMENT
# =================================================================

def setup_clean_directory(source_root: str, target_root: str):
    """Copie la structure des sous-dossiers vers le nouveau dossier cible."""
    print(f"Création du dossier de sortie sécurisé : {target_root}")
    
    # 1. Suppression de l'ancien dossier cible s'il existe
    if os.path.exists(target_root):
        print("⚠️ Suppression de l'ancien dossier de nettoyage...")
        shutil.rmtree(target_root)
        
    # 2. Copie des sous-dossiers pertinents
    for sub_dir in SUBDIRECTORIES:
        source_path = os.path.join(source_root, sub_dir)
        target_path = os.path.join(target_root, sub_dir)
        
        if os.path.isdir(source_path):
            print(f"   - Copie de /{sub_dir}...")
            shutil.copytree(source_path, target_path)
        else:
            print(f"   - Dossier source /{sub_dir} non trouvé. Ignoré.")

    print("Préparation du nouveau dataset terminée.")


def get_class_from_filename(filename: str) -> str:
    """Utilise la regex pour extraire le code de classe ou retourne 'unknown'."""
    basename = os.path.splitext(filename)[0]
    match = PATTERN_REGEX.search(basename)
    
    if match:
        return match.group(1)
    else:
        return 'unknown'


def delete_files_in_subdir(root_dir: str, sub_dir: str) -> Tuple[int, int]:
    """
    Parcourt un sous-dossier, identifie les images appartenant aux classes à supprimer 
    et supprime l'image et l'annotation XML associée (dans le dossier de COPIE).
    """
    images_dir = os.path.join(root_dir, sub_dir, 'images')
    annotations_dir = os.path.join(root_dir, sub_dir, 'Annotations')
    deleted_count = 0
    total_files_checked = 0

    if not os.path.isdir(images_dir):
        return 0, 0

    for filename in os.listdir(images_dir):
        if filename.lower().endswith(IMAGE_EXTENSIONS):
            total_files_checked += 1
            file_basename, ext = os.path.splitext(filename)
            
            # Déterminer la classe par le nom de fichier
            class_name = get_class_from_filename(filename)
            
            # Vérifier si la classe doit être supprimée
            is_minority = class_name in CLASSES_TO_DELETE
            is_unknown = class_name == 'unknown'
            
            if is_minority or is_unknown:
                
                image_path = os.path.join(images_dir, filename)
                xml_path = os.path.join(annotations_dir, file_basename + '.xml')
                
                # Suppression de l'image
                if os.path.exists(image_path):
                    os.remove(image_path)
                    
                # Suppression de l'annotation XML associée
                if os.path.exists(xml_path):
                    os.remove(xml_path)
                    
                deleted_count += 1
                print(f"  🗑️ Supprimé /{sub_dir}/{file_basename} (Classe: {class_name})") # Désactivé pour moins de logs

    return deleted_count, total_files_checked


# =================================================================
# 3. FONCTION PRINCIPALE
# =================================================================

def main(original_root_dir: str):
    max_workers = os.cpu_count() or 4
    
    # --- Étape 1 : Préparation de la copie ---
    target_root_dir = os.path.join(original_root_dir, CLEAN_ROOT_DIR)
    setup_clean_directory(original_root_dir, target_root_dir)

    print(f"\n--- Nettoyage par Multiprocessing dans {CLEAN_ROOT_DIR}... ---")
    print(f"Cœurs CPU utilisés : {max_workers}")

    # --- Étape 2 : Lancement du nettoyage sur la copie ---
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(delete_files_in_subdir, target_root_dir, sub_dir): sub_dir
            for sub_dir in SUBDIRECTORIES
        }

        total_deleted = 0
        
        # Récupérer les résultats
        for future in futures:
            sub_dir = futures[future]
            try:
                deleted, checked = future.result()
                total_deleted += deleted
                print(f"✅ Résultat pour /{sub_dir}: {deleted} paires Image/XML supprimées sur {checked} vérifiées.")
            except Exception as e:
                print(f"❌ Erreur lors du traitement du dossier /{sub_dir}: {e}")

    print("\n==================================================")
    print("✅ NETTOYAGE TERMINÉ !")
    print(f"Nouveau dataset 'clean' disponible dans le dossier : ./{CLEAN_ROOT_DIR}")
    print(f"Total des paires Image/XML supprimées : {total_deleted}")
    print(f"Dataset original : Non modifié.")
    print("==================================================")


if __name__ == '__main__':
    current_root_dir = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\dataset_pascal_voc"

    main(current_root_dir)