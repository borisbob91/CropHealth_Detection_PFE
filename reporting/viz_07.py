import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Données des performances par classe
data_classes = pd.DataFrame({
    "Classe": [
        "A. flava", "B. tabaci", "Coccinelle", "Degat Jassides", "Dysdercus spp",
        "Earias spp", "Effet phyto", "G. spodoptera", "H. armigera", "Jasside",
        "Larve coccinelle", "Larve syrphe", "P. gossypiella", "Puceron",
        "S. derogata", "S. frugiperda", "Scarabees"
    ],
    "Précision": [
        0.98, 0.964, 0.814, 0.788, 0.978,
        0.981, 0.603, 0.996, 0.986, 0.855,
        0.941, 0.959, 0.925, 0.505,
        0.905, 0.933, 0.995
    ],
    "Rappel": [
        0.987, 0.941, 0.872, 0.682, 0.98,
        0.966, 0.738, 1.000, 0.989, 0.805,
        0.991, 1.000, 0.967, 0.360,
        0.940, 1.000, 0.880
    ],
    "mAP@50": [
        0.978, 0.966, 0.870, 0.739, 0.974,
        0.992, 0.698, 0.995, 0.994, 0.807,
        0.992, 0.995, 0.969, 0.386,
        0.920, 0.984, 0.886
    ],
    "F1_score": [
        0.983487544, 0.952361155, 0.842002372, 0.731178231, 0.978998979,
        0.973442219, 0.663704698, 0.997995992, 0.987497722, 0.824515152,
        0.965353002, 0.979070955, 0.945533827, 0.420346821,
        0.922168022, 0.965338852, 0.933973333
    ]
})

# Trier par F1-score décroissant
data_sorted = data_classes.sort_values('F1_score', ascending=False)
classes = data_sorted['Classe'].tolist()
precision_scores = (data_sorted['Précision'] * 100).tolist()  # Conversion en pourcentage
rappel_scores = (data_sorted['Rappel'] * 100).tolist()

# Configuration du graphique
plt.figure(figsize=(18, 10))

x = np.arange(len(classes))
largeur = 0.35

# Couleurs de votre charte
couleur_precision = '#2E86AB'  # Bleu pour Précision
couleur_rappel = '#A23B72'     # Rose pour Rappel

# Création des barres groupées
bars_precision = plt.bar(x - largeur/2, precision_scores, largeur,
                         label='Précision (P)', 
                         color=couleur_precision, 
                         edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

bars_rappel = plt.bar(x + largeur/2, rappel_scores, largeur,
                      label='Rappel (R)', 
                      color=couleur_rappel, 
                      edgecolor='black', linewidth=2, alpha=0.9, zorder=3)

# Personnalisation
plt.title('Précision et Rappel par Classe d\'Insectes', 
          fontsize=24, fontweight='bold', pad=30)
plt.ylabel('Score (%)', fontsize=20, fontweight='bold')
plt.xlabel('Classes d\'Insectes (triées par F1-score)', fontsize=20, fontweight='bold')

plt.xticks(x, classes, fontsize=13, rotation=45, ha='right')
plt.ylim(30, 105)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres - Précision
for bar, valeur in zip(bars_precision, precision_scores):
    height = bar.get_height()
    # Ajuster position verticale selon la valeur
    if height > 90:
        y_pos = height - 3
        va_pos = 'top'
        color = 'white'
    else:
        y_pos = height + 1
        va_pos = 'bottom'
        color = '#2E86AB'
    
    plt.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{valeur:.1f}', ha='center', va=va_pos,
             fontsize=11, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.2', 
                      facecolor=couleur_precision if height > 90 else 'white',
                      alpha=0.9, edgecolor='black'))

# Valeurs sur les barres - Rappel
for bar, valeur in zip(bars_rappel, rappel_scores):
    height = bar.get_height()
    # Ajuster position verticale selon la valeur
    if height > 90:
        y_pos = height - 3
        va_pos = 'top'
        color = 'white'
    else:
        y_pos = height + 1
        va_pos = 'bottom'
        color = '#A23B72'
    
    plt.text(bar.get_x() + bar.get_width()/2., y_pos,
             f'{valeur:.1f}', ha='center', va=va_pos,
             fontsize=11, fontweight='bold', color=color,
             bbox=dict(boxstyle='round,pad=0.2', 
                      facecolor=couleur_rappel if height > 90 else 'white',
                      alpha=0.9, edgecolor='black'))

# Légende en haut, centrée
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), 
           ncol=2, fontsize=16, framealpha=0.9, frameon=True)

# Ligne horizontale pour la moyenne de Précision
mean_precision = np.mean(precision_scores)
plt.axhline(y=mean_precision, color=couleur_precision, linestyle='--', 
            linewidth=2.5, alpha=0.7, zorder=1)
plt.text(len(classes) - 0.5, mean_precision + 1,
         f'P moy: {mean_precision:.1f}%',
         fontsize=12, fontweight='bold', color=couleur_precision,
         ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor=couleur_precision, alpha=0.9))

# Ligne horizontale pour la moyenne de Rappel
mean_rappel = np.mean(rappel_scores)
plt.axhline(y=mean_rappel, color=couleur_rappel, linestyle='--', 
            linewidth=2.5, alpha=0.7, zorder=1)
plt.text(len(classes) - 0.5, mean_rappel - 1,
         f'R moy: {mean_rappel:.1f}%',
         fontsize=12, fontweight='bold', color=couleur_rappel,
         ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor=couleur_rappel, alpha=0.9))

# Ligne horizontale pour le seuil de 90%
plt.axhline(y=90, color='#2E8B57', linestyle=':', 
            linewidth=2, alpha=0.6, zorder=1)
plt.text(0, 91, 'Seuil 90%',
         fontsize=11, fontweight='bold', color='#2E8B57',
         ha='left', va='bottom',
         bbox=dict(boxstyle='round,pad=0.2', 
                  facecolor='white', 
                  edgecolor='#2E8B57', alpha=0.8))

# Annoter les classes avec équilibre P/R
for i, (p, r) in enumerate(zip(precision_scores, rappel_scores)):
    # Calculer la différence absolue
    diff = abs(p - r)
    
    # Si P et R sont très proches (équilibre)
    if diff < 5:  # Différence inférieure à 5%
        # Choisir la position Y (le plus haut des deux)
        y_pos = max(p, r) + 3
        
        # Choisir la couleur selon les scores
        if p > 90 and r > 90:
            color = '#2E8B57'  # Vert - excellent équilibre
            marker = '★'
            label = 'Excellent'
        elif p > 80 and r > 80:
            color = '#FF8C00'  # Orange - bon équilibre
            marker = '✓'
            label = 'Bon équilibre'
        else:
            color = '#6495ED'  # Bleu clair - équilibre moyen
            marker = '≈'
            label = 'Équilibre'
        
        plt.annotate(marker, 
                     xy=(i, y_pos - 3),
                     xytext=(i, y_pos),
                     ha='center', va='bottom',
                     fontsize=14, fontweight='bold', color=color,
                     bbox=dict(boxstyle='round,pad=0.2', 
                              facecolor='white', 
                              edgecolor=color, alpha=0.9))

# Mettre en évidence les cas extrêmes
# Puceron - très faible rappel
puceron_idx = classes.index('Puceron')
plt.annotate('Rappel très faible', 
             xy=(puceron_idx, rappel_scores[puceron_idx]),
             xytext=(puceron_idx, 25),
             ha='center', va='top',
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', lw=1.5, color='red'),
             bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor='lightcoral', 
                      alpha=0.8, edgecolor='red'))

# G. spodoptera - rappel parfait
gspodoptera_idx = classes.index('G. spodoptera')
plt.annotate('Rappel parfait', 
             xy=(gspodoptera_idx, rappel_scores[gspodoptera_idx]),
             xytext=(gspodoptera_idx, 102),
             ha='center', va='bottom',
             fontsize=11, fontweight='bold',
             arrowprops=dict(arrowstyle='->', lw=1.5, color='green'),
             bbox=dict(boxstyle='round,pad=0.3', 
                      facecolor='lightgreen', 
                      alpha=0.8, edgecolor='green'))

# Ajustements finaux
plt.tight_layout()
plt.subplots_adjust(top=0.90, bottom=0.12)

plt.show()