import os
from pathlib import Path
import pandas as pd
from collections import defaultdict

def count_images_by_class(root_path):
    """
    Compte le nombre d'images par classe dans chaque ensemble (train, val, test)
    
    Args:
        root_path: Chemin racine du dataset
    """
    root = Path(root_path)
    
    # Extensions d'images supportées
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
    
    # Structure pour stocker les résultats
    data = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    
    # Parcourir les ensembles (train, val, test)
    for dataset in ['train', 'val', 'test']:
        dataset_path = root / dataset
        
        if not dataset_path.exists():
            print(f"Attention: Le dossier {dataset} n'existe pas")
            continue
        
        # Parcourir les classes dans chaque ensemble
        for class_folder in dataset_path.iterdir():
            if class_folder.is_dir():
                class_name = class_folder.name
                
                # Compter les images dans cette classe
                image_count = sum(
                    1 for f in class_folder.iterdir() 
                    if f.is_file() and f.suffix.lower() in image_extensions
                )
                
                data[class_name][dataset] = image_count
    
    # Convertir en DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.index.name = 'Classe'
    
    # Ajouter une colonne Total
    df['Total'] = df['train'] + df['val'] + df['test']
    
    # Trier par nom de classe
    df = df.sort_index()
    
    # Ajouter une ligne de total
    totals = df.sum()
    totals.name = 'TOTAL'
    df = pd.concat([df, pd.DataFrame([totals])])
    
    return df

def main():
    # Demander le chemin du dataset
    root_path = input("Entrez le chemin racine du dataset: ").strip()
    
    if not os.path.exists(root_path):
        print(f"Erreur: Le chemin {root_path} n'existe pas")
        return
    
    print(f"\nAnalyse du dataset: {root_path}")
    print("-" * 50)
    
    # Compter les images
    df = count_images_by_class(root_path)
    
    # Afficher les résultats
    print("\nRésultats:")
    print(df.to_string())
    
    # Sauvegarder dans Excel
    output_file = Path(root_path) / "statistiques_images.xlsx"
    df.to_excel(output_file)
    
    print(f"\n✓ Fichier Excel créé: {output_file}")

if __name__ == "__main__":
    main()