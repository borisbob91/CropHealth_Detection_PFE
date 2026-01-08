import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# ============================================
# PARAMÈTRES DE CONFIGURATION
# ============================================
FICHIER_EXCEL = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\state\instances_augment.xlsx"

# CONFIGURATION DU GRAPHIQUE
ORIENTATION = "horizontal"  # "horizontal" ou "vertical"
TRI_DECROISSANT = True      # True pour trier par ordre décroissant, False pour ordre alphabétique
AFFICHER_POURCENTAGES = True  # True pour afficher les pourcentages sur les barres

# Style du graphique
COULEUR_IMAGES = '#A23B72'
TAILLE_FIGURE_H = (12, 10)  # Pour horizontal
TAILLE_FIGURE_V = (16, 8)   # Pour vertical

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
try:
    df = pd.read_excel(FICHIER_EXCEL)
    print("\n✓ Données chargées avec succès!")
    print(df.head())
    
except FileNotFoundError:
    print(f"\n⚠ Fichier non trouvé. Utilisation des données de démonstration...")
    data = {
        'Classe': ['A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp',
                  'Earias spp', 'Effet phyto', 'G. spodoctera', 'H. amirgera', 'Jasside',
                  'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron',
                  'S. derogata', 'S. frugiperda', 'Scarabees'],
        'total_objets': [51, 176, 297, 332, 110, 355, 237, 58, 398, 1797, 372, 61, 67, 1186, 655, 222, 95],
        'total_img': [51, 30, 282, 152, 113, 354, 205, 58, 398, 199, 266, 35, 67, 276, 552, 223, 92]
    }
    df = pd.DataFrame(data)

# ============================================
# PRÉPARATION DES DONNÉES
# ============================================
# Tri des données si demandé
if TRI_DECROISSANT:
    # Trier par nombre d'images décroissant
    df = df.sort_values('total_img', ascending=False)
else:
    # Ordre alphabétique
    df = df.sort_values('Classe')

# Extraction des données
classes = df['Classe'].tolist()
total_img = df['total_img'].tolist()

# Calcul des pourcentages
total_img_global = sum(total_img)
pourcentage_img = [img/total_img_global * 100 for img in total_img]

# ============================================
# CRÉATION DU GRAPHIQUE
# ============================================
if ORIENTATION == "horizontal":
    # GRAPHIQUE HORIZONTAL
    fig, ax = plt.subplots(figsize=TAILLE_FIGURE_H)
    
    y = np.arange(len(classes))
    hauteur = 0.6
    
    # Création des barres horizontales
    bars = ax.barh(y, pourcentage_img, hauteur, 
                   color=COULEUR_IMAGES, edgecolor='black', 
                   linewidth=1.2, zorder=3, alpha=0.9)
    
    # Configuration des axes
    ax.set_xlabel('Pourcentage (%)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Classes', fontsize=13, fontweight='bold')
    ax.set_yticks(y)
    ax.set_yticklabels(classes, fontsize=10)
    ax.set_xlim(0, max(pourcentage_img) * 1.12)
    
    # Format pourcentage
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    
    # Grille
    ax.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)
    
    # Ajouter les pourcentages
    if AFFICHER_POURCENTAGES:
        for bar, pct in zip(bars, pourcentage_img):
            width = bar.get_width()
            if width > 1.0:  # N'afficher que si la barre est assez grande
                y_pos = bar.get_y() + bar.get_height()/2.
                ax.text(width + 0.2, y_pos, f"{pct:.1f}%", 
                       ha='left', va='center', fontsize=9, fontweight='bold',
                       color=COULEUR_IMAGES)
    
else:
    # GRAPHIQUE VERTICAL
    fig, ax = plt.subplots(figsize=TAILLE_FIGURE_V)
    
    x = np.arange(len(classes))
    largeur = 0.6
    
    # Création des barres verticales
    bars = ax.bar(x, pourcentage_img, largeur, 
                  color=COULEUR_IMAGES, edgecolor='black', 
                  linewidth=1.2, zorder=3, alpha=0.9)
    
    # Configuration des axes
    ax.set_ylabel('Pourcentage (%)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Classes', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
    ax.set_ylim(0, max(pourcentage_img) * 1.12)
    
    # Format pourcentage
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))
    
    # Grille
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    
    # Ajouter les pourcentages
    if AFFICHER_POURCENTAGES:
        for bar, pct in zip(bars, pourcentage_img):
            height = bar.get_height()
            if height > 1.0:  # N'afficher que si la barre est assez grande
                x_pos = bar.get_x() + bar.get_width()/2.
                ax.text(x_pos, height + 0.2, f"{pct:.1f}%", 
                       ha='center', va='bottom', fontsize=9, fontweight='bold',
                       color=COULEUR_IMAGES)

# ============================================
# FINALISATION
# ============================================
# Ajuster les marges pour que tout soit bien visible
plt.tight_layout()

# Afficher les statistiques
print(f"\n{'='*60}")
print("CONFIGURATION DU GRAPHIQUE:")
print(f"{'='*60}")
print(f"Orientation: {ORIENTATION}")
print(f"Tri décroissant: {TRI_DECROISSANT}")
print(f"Afficher pourcentages: {AFFICHER_POURCENTAGES}")
print(f"\n{'='*60}")
print("STATISTIQUES:")
print(f"{'='*60}")
print(f"Total images: {total_img_global:,}")
print(f"Nombre de classes: {len(classes)}")
print(f"Moyenne images/classe: {total_img_global/len(classes):.1f}")
print(f"{'='*60}")
print("\nTop 5 des classes par nombre d'images:")
top5 = df.nlargest(5, 'total_img')[['Classe', 'total_img']]
for idx, row in top5.iterrows():
    pct = (row['total_img']/total_img_global)*100
    print(f"  {row['Classe']}: {row['total_img']} images ({pct:.1f}%)")
print(f"{'='*60}\n")

# Sauvegarder le graphique (optionnel)
# plt.savefig('distribution_images.png', dpi=300, bbox_inches='tight')

plt.show()