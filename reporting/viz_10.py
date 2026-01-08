import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# Demander le fichier Excel à l'utilisateur
fichier_excel = r"C:\Users\BorisBob\Desktop\detection\dataset_split\label_studio\state\instances_count.xlsx"

# Lire les données depuis Excel
try:
    df = pd.read_excel(fichier_excel)
    print("\nDonnées chargées avec succès!")
    print(df.head())
    
except FileNotFoundError:
    print(f"\nErreur: Le fichier '{fichier_excel}' n'a pas été trouvé.")
    print("Création de données de démonstration...")
    
    # Données de démonstration
    data = {
        'Classe': ['A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 'Dysdercus spp',
                  'Earias spp', 'Effet phyto', 'G. spodoctera', 'H. amirgera', 'Jasside',
                  'Larve coccinelle', 'Larve syrphe', 'P. gossypiella', 'Puceron',
                  'S. derogata', 'S. frugiperda', 'Scarabees'],
        'total_objets': [51, 176, 297, 332, 110, 355, 237, 58, 398, 1797, 372, 61, 67, 1186, 655, 222, 95],
        'total_img': [51, 30, 282, 152, 113, 354, 205, 58, 398, 199, 266, 35, 67, 276, 552, 223, 92]
    }
    df = pd.DataFrame(data)

# Extraire les données
classes = df['Classe'].tolist()
total_objets = df['total_objets'].tolist()
total_img = df['total_img'].tolist()

# Calculer les totaux pour les pourcentages
total_objets_global = sum(total_objets)
total_img_global = sum(total_img)

# Calculer les pourcentages
pourcentage_objets = [obj/total_objets_global * 100 for obj in total_objets]
pourcentage_img = [img/total_img_global * 100 for img in total_img]

# ============================================
# VERSION 1 : STYLE BARRES GROUPÉES (POURCENTAGES)
# ============================================

fig1, ax1 = plt.subplots(figsize=(18, 10))

x = np.arange(len(classes))
largeur = 0.35

# Création des barres groupées
bars1 = ax1.bar(x - largeur/2, pourcentage_objets, largeur, 
                label='% Instances', color='#2E86AB', edgecolor='black', 
                linewidth=1.5, zorder=3, alpha=0.9)
bars2 = ax1.bar(x + largeur/2, pourcentage_img, largeur, 
                label='% Images', color='#A23B72', edgecolor='black', 
                linewidth=1.5, zorder=3, alpha=0.9)

# Personnalisation
ax1.set_ylabel('Pourcentage (%)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=10, ha='right', fontsize=11)
ax1.set_ylim(0, max(pourcentage_objets + pourcentage_img) * 1.15)

# Format pourcentage sur l'axe Y
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Fonction pour placer les textes intelligemment
def ajouter_textes_pourcentages(bars, valeurs_pct, valeurs_abs, ax, est_objets=True):
    """Ajoute les textes sans chevauchement pour les pourcentages"""
    positions_utilisees = []
    
    for bar, pct, abs_val in zip(bars, valeurs_pct, valeurs_abs):
        x_pos = bar.get_x() + bar.get_width()/2.
        height = bar.get_height()
        
        # Texte avec pourcentage uniquement
        texte = f"{pct:.1f}%"
        
        # Position verticale
        y_pos = height + 0.3
        
        # Vérifier les chevauchements
        chevauchement = False
        for pos in positions_utilisees:
            if abs(pos[0] - x_pos) < 0.25 and abs(pos[1] - y_pos) < 1.0:
                chevauchement = True
                break
        
        # Ajuster la position si chevauchement
        if chevauchement:
            y_pos = height + 0.8
        
        # Couleur du texte adaptée
        color = '#2E86AB' if est_objets else '#A23B72'
        
        # Ajouter le texte
        if height > 0.5:
            ax.text(x_pos, y_pos, texte, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=color,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                             edgecolor=color, alpha=0.9, linewidth=1))
        
        positions_utilisees.append((x_pos, y_pos))

# Ajouter les textes
ajouter_textes_pourcentages(bars1, pourcentage_objets, total_objets, ax1, est_objets=True)
ajouter_textes_pourcentages(bars2, pourcentage_img, total_img, ax1, est_objets=False)

# Légende réduite et placée en haut à droite dans la zone du graphique
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, 
           edgecolor='gray', fancybox=True, shadow=True)

# Ajuster la disposition
plt.tight_layout()
plt.show()
