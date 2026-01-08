import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker

# ============================================
# DONNÉES DE PERFORMANCES
# ============================================
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

# Convertir en pourcentages
data_classes['Précision'] = data_classes['Précision'] * 100
data_classes['Rappel'] = data_classes['Rappel'] * 100
data_classes['mAP@50'] = data_classes['mAP@50'] * 100
data_classes['F1_score'] = data_classes['F1_score'] * 100

# Extraire les données (GARDER L'ORDRE ORIGINAL)
classes = data_classes['Classe'].tolist()
precision = data_classes['Précision'].tolist()
rappel = data_classes['Rappel'].tolist()
map50 = data_classes['mAP@50'].tolist()
f1_score = data_classes['F1_score'].tolist()

# Fonction pour attribuer une couleur selon le seuil
def get_color(value):
    if value >= 80:
        return '#06A77D'  # Vert
    elif value >= 50:
        return '#F77F00'  # Orange
    else:
        return '#E63946'  # Rouge

# ============================================
# GRAPHIQUE 1 : PRÉCISION ET RAPPEL PAR CLASSE (VERTICAL)
# ============================================

fig1, ax1 = plt.subplots(figsize=(20, 10))

x = np.arange(len(classes))
largeur = 0.45

# Couleurs distinctes pour Précision et Rappel
couleur_precision = "#0679AA"
couleur_rappel = '#E63946'

# Création des barres verticales
bars_p = ax1.bar(x - largeur/2, precision, largeur, 
                 label='Précision', color=couleur_precision, 
                 edgecolor='black', linewidth=1.5, zorder=3, alpha=0.9)
bars_r = ax1.bar(x + largeur/2, rappel, largeur, 
                 label='Rappel', color=couleur_rappel, 
                 edgecolor='black', linewidth=1.5, zorder=3, alpha=0.9)

# Configuration des axes
ax1.set_ylabel('Performance (%)', fontsize=10)
ax1.set_xlabel('Classes', fontsize=10,)
ax1.set_xticks(x)
ax1.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
ax1.set_ylim(0, 110)

# Format pourcentage
ax1.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax1.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Ajouter les valeurs en pourcentage sur les barres (TEXTE VERTICAL)
for bar, val in zip(bars_p, precision):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width()/2.
    
    # Texte au-dessus de la barre en VERTICAL
    ax1.text(x_pos, height + 1, f"{val:.1f}%", 
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=couleur_precision, rotation=90,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                     edgecolor=couleur_precision, alpha=0.9, linewidth=1))

for bar, val in zip(bars_r, rappel):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width()/2.
    
    # Texte au-dessus de la barre en VERTICAL
    ax1.text(x_pos, height + 1, f"{val:.1f}%", 
            ha='center', va='bottom', fontsize=9, fontweight='bold',
            color=couleur_rappel, rotation=90,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                     edgecolor=couleur_rappel, alpha=0.9, linewidth=1))
# Ligne de référence à 80%
"""
ax1.axhline(y=80, color='black', linestyle=':', linewidth=2.5, alpha=0.6, zorder=1)
ax1.text(len(classes)-0.5, 81, '80%', ha='right', va='bottom',
        fontsize=11, color='black', fontweight='bold')
"""



# Légende
ax1.legend(loc='lower right', fontsize=13, framealpha=0.95,
          edgecolor='#F77F00', fancybox=True, shadow=True)

# Titre
ax1.set_title('Précision et Rappel par Classe', 
             fontsize=10, pad=25)

plt.tight_layout()
plt.show()

# ============================================
# GRAPHIQUE 2 : F1-SCORE PAR CLASSE (VERTICAL - COULEURS PAR SEUIL)
# ============================================

fig2, ax2 = plt.subplots(figsize=(20, 10))

# Attribuer les couleurs selon les seuils
couleurs_f1 = [get_color(val) for val in f1_score]

# Création des barres verticales
bars_f1 = ax2.bar(x, f1_score, 0.6, 
                  color=couleurs_f1, 
                  edgecolor='black', linewidth=1.5, zorder=3, alpha=0.9)

# Configuration des axes
ax2.set_ylabel('F1-Score (%)', fontsize=10)
ax2.set_xlabel('Classes', fontsize=10)
ax2.set_xticks(x)
ax2.set_xticklabels(classes, rotation=45, ha='right', fontsize=10)
ax2.set_ylim(0, 110)

# Format pourcentage
ax2.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax2.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Ajouter les valeurs en pourcentage sur les barres
for bar, val, couleur in zip(bars_f1, f1_score, couleurs_f1):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width()/2.
    
    # Texte au-dessus de la barre
    ax2.text(x_pos, height + 1, f"{val:.1f}%", 
            ha='center', va='bottom', fontsize=10,
            color=couleur,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=couleur, alpha=0.9, linewidth=1.5))

# Lignes de référence
ax2.axhline(y=80, color="#E4082C", linestyle=':', linewidth=2.5, alpha=0.6, zorder=1)
ax2.text(len(classes)-0.5, 81, '80%', ha='right', va='bottom',
        fontsize=11, color="#E4082C", fontweight='bold')

ax2.axhline(y=50, color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.6, zorder=1)
ax2.text(len(classes)-0.5, 51, '50%', ha='right', va='bottom',
        fontsize=11, color='#F77F00', fontweight='bold')

# Légende pour les seuils
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#06A77D', alpha=0.9, edgecolor='black', label='≥ 80% (Excellent)'),
    Patch(facecolor='#F77F00', alpha=0.9, edgecolor='black', label='50-80% (Moyen)'),
    Patch(facecolor='#E63946', alpha=0.9, edgecolor='black', label='< 50% (Faible)')
]
ax2.legend(handles=legend_elements, loc='lower right', fontsize=12, 
          framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)

# Titre
ax2.set_title('F1-Score par Classe', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# ============================================
# GRAPHIQUE 3 : mAP@50 PAR CLASSE (VERTICAL - COULEURS PAR SEUIL)
# ============================================

fig3, ax3 = plt.subplots(figsize=(20, 10))

# Attribuer les couleurs selon les seuils
couleurs_map = [get_color(val) for val in map50]

# Création des barres verticales
bars_map = ax3.bar(x, map50, 0.6, 
                   color=couleurs_map, 
                   edgecolor='black', linewidth=1.5, zorder=3, alpha=0.9)

# Configuration des axes
ax3.set_ylabel('mAP@50 (%)', fontsize=16, fontweight='bold')
ax3.set_xlabel('Classes', fontsize=16, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(classes, rotation=45, ha='right', fontsize=12)
ax3.set_ylim(0, 110)

# Format pourcentage
ax3.yaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax3.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)

# Ajouter les valeurs en pourcentage sur les barres
for bar, val, couleur in zip(bars_map, map50, couleurs_map):
    height = bar.get_height()
    x_pos = bar.get_x() + bar.get_width()/2.
    
    # Texte au-dessus de la barre
    ax3.text(x_pos, height + 1, f"{val:.1f}%", 
            ha='center', va='bottom', fontsize=10, fontweight='bold',
            color=couleur,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=couleur, alpha=0.9, linewidth=1.5))

# Lignes de référence
ax3.axhline(y=80, color="#04634A", linestyle=':', linewidth=2.5, alpha=0.6, zorder=1)
ax3.text(len(classes)-0.5, 81, '80%', ha='right', va='bottom',
        fontsize=11, color="#03644A", fontweight='bold')

ax3.axhline(y=50, color='#F77F00', linestyle=':', linewidth=2.5, alpha=0.6, zorder=1)
ax3.text(len(classes)-0.5, 51, '50%', ha='right', va='bottom',
        fontsize=11, color='#F77F00', fontweight='bold')

# Légende pour les seuils
legend_elements = [
    Patch(facecolor='#06A77D', alpha=0.9, edgecolor='black', label='≥ 80% (Excellent)'),
    Patch(facecolor='#F77F00', alpha=0.9, edgecolor='black', label='50-80% (Moyen)'),
    Patch(facecolor='#E63946', alpha=0.9, edgecolor='black', label='< 50% (Faible)')
]
ax3.legend(handles=legend_elements, loc='lower right', fontsize=12, 
          framealpha=0.95, edgecolor='gray', fancybox=True, shadow=True)

# Titre
ax3.set_title('mAP@50 par Classe', 
             fontsize=18, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# ============================================
# STATISTIQUES
# ============================================

print(f"\n{'='*90}")
print("PERFORMANCES PAR CLASSE (ordre original)")
print(f"{'='*90}")
print(f"{'CLASSE':<25} {'PRÉCISION':>12} {'RAPPEL':>12} {'F1-SCORE':>12} {'mAP@50':>12}")
print(f"{'-'*90}")

for classe, prec, rec, f1, map_val in zip(classes, precision, rappel, f1_score, map50):
    print(f"{classe:<25} {prec:>11.1f}% {rec:>11.1f}% {f1:>11.1f}% {map_val:>11.1f}%")

print(f"{'-'*90}")
print(f"{'MOYENNE':<25} {np.mean(precision):>11.1f}% {np.mean(rappel):>11.1f}% {np.mean(f1_score):>11.1f}% {np.mean(map50):>11.1f}%")
print(f"{'='*90}")

print(f"\n📊 RÉPARTITION PAR NIVEAU DE PERFORMANCE:")
print(f"\nF1-Score:")
excellent_f1 = len([x for x in f1_score if x >= 80])
moyen_f1 = len([x for x in f1_score if 50 <= x < 80])
faible_f1 = len([x for x in f1_score if x < 50])
print(f"  • Excellent (≥80%): {excellent_f1} classes")
print(f"  • Moyen (50-80%): {moyen_f1} classes")
print(f"  • Faible (<50%): {faible_f1} classes")

print(f"\nmAP@50:")
excellent_map = len([x for x in map50 if x >= 80])
moyen_map = len([x for x in map50 if 50 <= x < 80])
faible_map = len([x for x in map50 if x < 50])
print(f"  • Excellent (≥80%): {excellent_map} classes")
print(f"  • Moyen (50-80%): {moyen_map} classes")
print(f"  • Faible (<50%): {faible_map} classes")

print(f"{'='*90}\n")