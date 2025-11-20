#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour compter les objets par classe dans des annotations Pascal VOC XML
Usage: python count_voc_objects.py --input /chemin/vers/dossier --recursive
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter
import argparse
import json
import sys
from typing import Dict, List, Tuple, Optional


def count_objects_in_xml(xml_path: Path) -> Counter:
    """
    Compte les objets par classe dans un fichier XML Pascal VOC.
    
    Args:
        xml_path: Chemin vers le fichier XML
        
    Returns:
        Counter avec les noms de classe comme clés et les comptes comme valeurs
    """
    class_counter = Counter()
    
    try:
        # Parser le fichier XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Gérer les espaces de noms si présents
        namespace = {'voc': 'http://host.robots.ox.ac.uk/recipe'}
        
        # Trouver tous les objets
        objects = root.findall('.//object', namespace) or root.findall('.//object')
        
        for obj in objects:
            # Extraire le nom de la classe
            name_elem = obj.find('name', namespace) or obj.find('name')
            
            if name_elem is not None and name_elem.text:
                class_name = name_elem.text.strip()
                class_counter[class_name] += 1
                
    except ET.ParseError as e:
        print(f"❌ Erreur de parsing XML dans {xml_path}: {e}", file=sys.stderr)
    except Exception as e:
        print(f"❌ Erreur inattendue avec {xml_path}: {e}", file=sys.stderr)
        
    return class_counter


def count_objects_in_directory(
    input_dir: Path, 
    recursive: bool = False,
    file_pattern: str = "*.xml"
) -> Tuple[Counter, List[Path]]:
    """
    Parcourt un répertoire et compte les objets dans tous les fichiers XML.
    
    Args:
        input_dir: Répertoire à parcourir
        recursive: Si True, parcourt aussi les sous-répertoires
        file_pattern: Pattern de fichier (par défaut: *.xml)
        
    Returns:
        Tuple de (Counter des classes, liste des fichiers traités)
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Le répertoire {input_dir} n'existe pas")
    
    if not input_dir.is_dir():
        raise NotADirectoryError(f"{input_dir} n'est pas un répertoire")
    
    # Trouver les fichiers XML
    if recursive:
        xml_files = list(input_dir.rglob(file_pattern))
    else:
        xml_files = list(input_dir.glob(file_pattern))
    
    if not xml_files:
        print(f"⚠️ Aucun fichier {file_pattern} trouvé dans {input_dir}")
        return Counter(), []
    
    print(f"📁 Trouvé {len(xml_files)} fichier(s) XML")
    
    # Compter les objets
    total_counter = Counter()
    processed_files = []
    
    for xml_file in xml_files:
        file_counter = count_objects_in_xml(xml_file)
        
        if file_counter:
            total_counter.update(file_counter)
            processed_files.append(xml_file)
            
            # Afficher le résultat par fichier (optionnel)
            print(f"  📄 {xml_file.name}: {dict(file_counter)}")
    
    return total_counter, processed_files


def print_results(counter: Counter, total_files: int):
    """Affiche les résultats de manière formatée."""
    if not counter:
        print("\n❌ Aucun objet trouvé.")
        return
    
    print(f"\n" + "="*50)
    print(f"📊 RÉSULTATS ({total_files} fichiers traités)")
    print("="*50)
    
    # Calculer le total
    total_objects = sum(counter.values())
    
    # Afficher les résultats par classe
    for class_name, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_objects) * 100 if total_objects > 0 else 0
        print(f"{class_name:<20} : {count:>6} objets ({percentage:>5.1f}%)")
    
    print("-"*50)
    print(f"{'TOTAL':<20} : {total_objects:>6} objets")
    print(f"{'Fichiers':<20} : {total_files:>6} XML")


def save_results(counter: Counter, output_path: Path):
    """Sauvegarde les résultats en JSON et CSV."""
    # JSON
    json_path = output_path.with_suffix('.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(dict(counter), f, indent=2, ensure_ascii=False)
    print(f"\n💾 Résultats sauvegardés en JSON: {json_path}")
    
    # CSV
    csv_path = output_path.with_suffix('.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("class,count\n")
        for class_name, count in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{class_name},{count}\n")
    print(f"💾 Résultats sauvegardés en CSV: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Compter les objets par classe dans des annotations Pascal VOC XML'
    )
    parser.add_argument(
        '-i', '--input',
        type=Path,
        default='.',
        help='Répertoire contenant les fichiers XML (défaut: .)'
    )
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='Parcourir récursivement les sous-répertoires'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='*.xml',
        help='Pattern de fichier (défaut: *.xml)'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Chemin pour sauvegarder les résultats (sans extension)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Mode silencieux (pas de sortie par fichier)'
    )
    
    args = parser.parse_args()
    
    # Compter les objets
    try:
        counter, processed_files = count_objects_in_directory(
            args.input,
            recursive=args.recursive,
            file_pattern=args.pattern
        )
        
        # Afficher les résultats
        if not args.quiet:
            print_results(counter, len(processed_files))
        else:
            print(f"\n✅ Traités {len(processed_files)} fichiers, {sum(counter.values())} objets au total")
            for cls, cnt in sorted(counter.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {cnt}")
        
        # Sauvegarder si demandé
        if args.output:
            save_results(counter, args.output)
            
    except Exception as e:
        print(f"❌ Erreur: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    """
    # Compter dans le répertoire courant
python count_voc_objects.py

# Compter dans un répertoire spécifique
python count_voc_objects.py -i /chemin/vers/annotations

# Compter récursivement dans les sous-répertoires
python count_voc_objects.py -i /chemin/vers/dataset -r

# Sauvegarder les résultats
python count_voc_objects.py -i /chemin/vers/annotations -o resultats
    """
    main()