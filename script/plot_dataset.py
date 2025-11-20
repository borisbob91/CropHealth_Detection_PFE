import argparse
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_stacked_bars(classes, train, val, test, totals=None, title='Graphique Empilé (Stacked Bar): Distribution Train/Val/Test par Classe', figsize=(10, 6)):
    """
    Graphique en barres empilées : Train en bas, Val milieu, Test en haut.
    """
    x = np.arange(len(classes))
    widths = 0.8  # Largeur des barres
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x, train, widths, label='Train', color='lightblue')
    ax.bar(x, val, widths, bottom=train, label='Val', color='lightgreen')
    ax.bar(x, test, widths, bottom=np.array(train) + np.array(val), label='Test', color='lightcoral')
    
    # Ajouter les totaux au-dessus (si fournis)
    if totals is not None:
        for i, total in enumerate(totals):
            ax.text(x[i], total + 0.5, str(total), ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def plot_side_by_side_bars(classes, train, val, test, title='Graphique Côte à Côte (Grouped Bar): Distribution Train/Val/Test par Classe', figsize=(10, 6)):
    """
    Graphique en barres groupées (side-by-side) : Train, Val, Test côte à côte pour chaque classe.
    """
    x = np.arange(len(classes))
    width = 0.25  # Largeur de chaque barre
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width, train, width, label='Train', color='lightblue')
    ax.bar(x, val, width, label='Val', color='lightgreen')
    ax.bar(x + width, test, width, label='Test', color='lightcoral')
    
    # Ajouter les valeurs sur les barres
    for i, (t, v, te) in enumerate(zip(train, val, test)):
        ax.text(x[i] - width, t + 0.5, str(t), ha='center', va='bottom', fontsize=9)
        ax.text(x[i], v + 0.5, str(v), ha='center', va='bottom', fontsize=9)
        ax.text(x[i] + width, te + 0.5, str(te), ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Nombre d\'échantillons')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# Exécuter les deux plots
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot dataset split statistics from CSV", usage="python plot_dataset.py --csv split_statistics.csv")
    parser.add_argument('--csv', '-c', type=str, default='split_statistics.csv',
                        help='Path to the split_statistics CSV file (default: split_statistics.csv in CWD)')
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    # Nettoyer et préparer les données (ignorer la ligne TOTAL si elle est redondante)
    class_rows = df[df['Classe'].str.contains('TOTAL', na=False) == False]

    # Colonnes pour les splits
    splits = ['Train', 'Val', 'Test']
    classes = class_rows['Classe'].tolist()
    train_data = class_rows['Train'].tolist()
    val_data = class_rows['Val'].tolist()
    test_data = class_rows['Test'].tolist()

    # Calculer les totaux si besoin
    totals = class_rows['Total'].tolist() if 'Total' in class_rows.columns else None

    print("Données chargées:")
    print(class_rows)

    plot_stacked_bars(classes, train_data, val_data, test_data, totals=totals)
    plot_side_by_side_bars(classes, train_data, val_data, test_data)

    # Afficher les infos supplémentaires du CSV (si besoin)
    print("\nInfos supplémentaires du CSV:")
    additional_info = """
    Nombre de classes: 1
    Patterns détectés: IMG_R_AF
    Images sans labels: 0
    Labels sans images: 0
    Seed aléatoire: 42
    Méthode: Stratification par classe (pattern-based)
    """
    print(additional_info)