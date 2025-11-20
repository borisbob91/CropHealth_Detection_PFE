import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Structure de données - copier-coller directement
DATA = [
    {'class': 'Syllepte_derogata_larve', 'train': 447, 'val': 57, 'test': 55, 'total': 559},
    {'class': 'Helicoverpa_armigera(Notuelle)', 'train': 266, 'val': 34, 'test': 33, 'total': 333},
    {'class': 'Earias_spp_(chenille_epineuse)', 'train': 247, 'val': 32, 'test': 30, 'total': 309},
    {'class': 'Coccinelles_spp', 'train': 231, 'val': 30, 'test': 28, 'total': 289},
    {'class': 'Larves_coccinelles', 'train': 217, 'val': 28, 'test': 27, 'total': 272},
    {'class': 'Aphis_gossypii_(pucerons)', 'train': 207, 'val': 27, 'test': 25, 'total': 259},
    {'class': 'Jassides', 'train': 160, 'val': 20, 'test': 20, 'total': 200},
    {'class': 'Jassides_attaques', 'train': 144, 'val': 18, 'test': 18, 'total': 180},
    {'class': 'Spodoctera_frugiperda', 'train': 124, 'val': 17, 'test': 15, 'total': 156},
    {'class': 'Dysdercus_spp_(punaise)', 'train': 90, 'val': 12, 'test': 11, 'total': 113},
    {'class': 'Scarabees_neutre', 'train': 75, 'val': 10, 'test': 9, 'total': 94},
    {'class': 'Pectinophora_gossypiella(ver_rose)', 'train': 53, 'val': 8, 'test': 6, 'total': 67},
    {'class': 'genre_spodoctora', 'train': 46, 'val': 7, 'test': 5, 'total': 58},
    {'class': 'Anomis_flava_(arpenteuse)', 'train': 37, 'val': 6, 'test': 4, 'total': 47},
    {'class': 'larve_syphe(predacteur)', 'train': 28, 'val': 4, 'test': 3, 'total': 35},
    {'class': 'symphe', 'train': 17, 'val': 3, 'test': 2, 'total': 22},
    {'class': 'fourmie_noire', 'train': 16, 'val': 3, 'test': 2, 'total': 21},
    {'class': 'Effet_herbicides', 'train': 169, 'val': 22, 'test': 21, 'total': 212},
    {'class': 'plainte_saine', 'train': 172, 'val': 23, 'test': 21, 'total': 216},
    {'class': 'Bemisia tabaci_(Mouche_blanche)', 'train': 14, 'val': 3, 'test': 1, 'total': 18},
    {'class': 'Syllepte_derogata_adulte', 'train': 20, 'val': 4, 'test': 2, 'total': 26},
    {'class': 'classX', 'train': 11, 'val': 2, 'test': 1, 'total': 14},
    {'class': 'punaisse_01', 'train': 11, 'val': 2, 'test': 1, 'total': 14},
    {'class': 'punaisse_02', 'train': 10, 'val': 2, 'test': 1, 'total': 13},
    {'class': 'puniase_04', 'train': 8, 'val': 1, 'test': 1, 'total': 10},
    {'class': 'Anomis_flava_attaque', 'train': 8, 'val': 1, 'test': 1, 'total': 10},
    {'class': 'puniase_03', 'train': 3, 'val': 1, 'test': 0, 'total': 4},
]

def plot_class_distribution(data=DATA, figsize=(14, 10), save_path=None):
    """
    Affiche un graphique des classes triées par nombre total d'exemples (décroissant).
    
    Args:
        data: Liste des dictionnaires de classes
        figsize: Taille de la figure (largeur, hauteur)
        save_path: Chemin pour sauvegarder l'image (optionnel)
    """
    # Créer DataFrame
    df = pd.DataFrame(data)
    
    # Trier par total décroissant
    df_sorted = df.sort_values('total', ascending=True)  # True pour barh (ordonnée)
    
    # Créer la figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Couleur dégradée selon le total
    colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
    
    # Bar plot horizontal
    bars = ax.barh(df_sorted['class'], df_sorted['total'], color=colors)
    
    # Ajouter les valeurs sur les barres
    for bar, total in zip(bars, df_sorted['total']):
        width = bar.get_width()
        ax.text(width + max(df_sorted['total']) * 0.01, bar.get_y() + bar.get_height()/2, 
                str(total), ha='left', va='center', fontsize=9, fontweight='bold')
    
    # Personnalisation
    ax.set_xlabel('Nombre d\'exemples', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classes', fontsize=12, fontweight='bold')
    ax.set_title('Distribution des classes par nombre d\'exemples\n(Ordre décroissant)', 
                 fontsize=14, fontweight='bold', pad=20)
    
    # Grille verticale
    ax.grid(axis='x', alpha=0.3)
    
    # Ajuster les ticks
    plt.yticks(rotation=0, fontsize=10)
    plt.xticks(fontsize=10)
    
    # Retirer les bordures inutiles
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Sauvegarder si demandé
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    
    plt.show()
    return fig


def plot_split_distribution(data=DATA, figsize=(16, 10), save_path=None):
    """
    Version avancée avec visualisation détaillée train/val/test par classe.
    """
    df = pd.DataFrame(data)
    df_sorted = df.sort_values('total', ascending=True)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Position des barres
    y_pos = np.arange(len(df_sorted))
    height = 0.6
    
    # Barres empilées
    bars1 = ax.barh(y_pos, df_sorted['train'], height, label='Train', color='#4CAF50')
    bars2 = ax.barh(y_pos, df_sorted['val'], height, 
                     left=df_sorted['train'], label='Val', color='#FFC107')
    bars3 = ax.barh(y_pos, df_sorted['test'], height,
                     left=df_sorted['train'] + df_sorted['val'], label='Test', color='#F44336')
    
    # Annotations des totaux
    for i, total in enumerate(df_sorted['total']):
        ax.text(total + max(df_sorted['total']) * 0.01, i, 
                f'{total}', va='center', ha='left', fontweight='bold', fontsize=9)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['class'])
    ax.set_xlabel('Nombre d\'exemples', fontweight='bold')
    ax.set_title('Distribution détaillée (Train/Val/Test) par classe\nOrdre décroissant', 
                 fontweight='bold', fontsize=14, pad=20)
    
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=0.3)
    
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Graphique sauvegardé: {save_path}")
    
    plt.show()
    return fig


# ============================================================
# EXECUTION
# ============================================================

if __name__ == "__main__":
    print("Génération du graphique principal...")
    plot_class_distribution(save_path="distribution_classes.png")
    
    print("\nGénération du graphique détaillé...")
    plot_split_distribution(save_path="distribution_split.png")