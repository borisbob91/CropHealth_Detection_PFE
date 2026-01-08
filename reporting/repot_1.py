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

# Extraire les données (GARDER L'ORDRE ORIGINAL)
classes = data_classes['Classe'].tolist()
precision = data_classes['Précision'].tolist()
rappel = data_classes['Rappel'].tolist()
map50 = data_classes['mAP@50'].tolist()

# ============================================
# GRAPHIQUE 1 : PRÉCISION ET RAPPEL PAR CLASSE
# ============================================

fig1, ax1 = plt.subplots(figsize=(16, 10))

y = np.arange(len(classes))
hauteur = 0.35

# Couleurs distinctes pour Précision et Rappel
couleur_precision = '#2E86AB'
couleur_rappel = '#E63946'

# Création des barres horizontales
bars_p = ax1.barh(y + hauteur/2, precision, hauteur, 
                  label='Précision', color=couleur_precision, 
                  edgecolor='black', linewidth=1.2, zorder=3, alpha=0.9)
bars_r = ax1.barh(y - hauteur/2, rappel, hauteur, 
                  label='Rappel', color=couleur_rappel, 
                  edgecolor='black', linewidth=1.2, zorder=3, alpha=0.9)

# Configuration des axes
ax1.set_xlabel('Performance (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Classes', fontsize=14, fontweight='bold')
ax1.set_yticks(y)
ax1.set_yticklabels(classes, fontsize=11)
ax1.set_xlim(0, 105)

# Format pourcentage
ax1.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax1.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

# Ajouter les valeurs en pourcentage sur les barres
for bar, val in zip(bars_p, precision):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height()/2.
    
    # Position du texte
    if width > 15:  # Si la barre est assez large
        x_pos = width / 2
        color = 'white'
        ha = 'center'
    else:
        x_pos = width + 1
        color = couleur_precision
        ha = 'left'
    
    ax1.text(x_pos, y_pos, f"{val:.1f}%", 
            ha=ha, va='center', fontsize=9, fontweight='bold',
            color=color)

for bar, val in zip(bars_r, rappel):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height()/2.
    
    # Position du texte
    if width > 15:
        x_pos = width / 2
        color = 'white'
        ha = 'center'
    else:
        x_pos = width + 1
        color = couleur_rappel
        ha = 'left'
    
    ax1.text(x_pos, y_pos, f"{val:.1f}%", 
            ha=ha, va='center', fontsize=9, fontweight='bold',
            color=color)

# Ligne de référence à 80%
ax1.axvline(x=80, color='gray', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
ax1.text(80, len(classes)-0.5, '80%', ha='center', va='bottom',
        fontsize=10, color='gray', fontweight='bold')

# Légende
ax1.legend(loc='lower right', fontsize=12, framealpha=0.95,
          edgecolor='gray', fancybox=True, shadow=True)

# Titre
ax1.set_title('Précision et Rappel par Classe', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# ============================================
# GRAPHIQUE 2 : mAP@50 PAR CLASSE
# ============================================

fig2, ax2 = plt.subplots(figsize=(16, 10))

# Couleur pour mAP@50
couleur_map = '#06A77D'

# Création des barres horizontales
bars_map = ax2.barh(y, map50, 0.6, 
                    color=couleur_map, 
                    edgecolor='black', linewidth=1.2, zorder=3, alpha=0.9)

# Configuration des axes
ax2.set_xlabel('mAP@50 (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Classes', fontsize=14, fontweight='bold')
ax2.set_yticks(y)
ax2.set_yticklabels(classes, fontsize=11)
ax2.set_xlim(0, 105)

# Format pourcentage
ax2.xaxis.set_major_formatter(ticker.PercentFormatter(decimals=0))

# Grille
ax2.grid(axis='x', linestyle='--', alpha=0.4, zorder=0)

# Ajouter les valeurs en pourcentage sur les barres
for bar, val in zip(bars_map, map50):
    width = bar.get_width()
    y_pos = bar.get_y() + bar.get_height()/2.
    
    # Position du texte
    if width > 15:  # Si la barre est assez large
        x_pos = width / 2
        color = 'white'
        ha = 'center'
    else:
        x_pos = width + 1
        color = couleur_map
        ha = 'left'
    
    ax2.text(x_pos, y_pos, f"{val:.1f}%", 
            ha=ha, va='center', fontsize=9, fontweight='bold',
            color=color)

# Ligne de référence à 80%
ax2.axvline(x=80, color='gray', linestyle=':', linewidth=2, alpha=0.6, zorder=1)
ax2.text(80, len(classes)-0.5, '80%', ha='center', va='bottom',
        fontsize=10, color='gray', fontweight='bold')

# Titre
ax2.set_title('mAP@50 par Classe', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# ============================================
# STATISTIQUES
# ============================================

print(f"\n{'='*80}")
print("PERFORMANCES PAR CLASSE (ordre original)")
print(f"{'='*80}")
print(f"{'CLASSE':<25} {'PRÉCISION':>12} {'RAPPEL':>12} {'mAP@50':>12}")
print(f"{'-'*80}")

for classe, prec, rec, map_val in zip(classes, precision, rappel, map50):
    print(f"{classe:<25} {prec:>11.1f}% {rec:>11.1f}% {map_val:>11.1f}%")

print(f"{'-'*80}")
print(f"{'MOYENNE':<25} {np.mean(precision):>11.1f}% {np.mean(rappel):>11.1f}% {np.mean(map50):>11.1f}%")
print(f"{'='*80}")

print(f"\n📊 STATISTIQUES GLOBALES:")
print(f"  • Précision moyenne: {np.mean(precision):.1f}%")
print(f"  • Rappel moyen: {np.mean(rappel):.1f}%")
print(f"  • mAP@50 moyen: {np.mean(map50):.1f}%")

print(f"\n🏆 MEILLEURES PERFORMANCES:")
best_precision = max(zip(classes, precision), key=lambda x: x[1])
best_rappel = max(zip(classes, rappel), key=lambda x: x[1])
best_map = max(zip(classes, map50), key=lambda x: x[1])

print(f"  • Meilleure précision: {best_precision[0]} ({best_precision[1]:.1f}%)")
print(f"  • Meilleur rappel: {best_rappel[0]} ({best_rappel[1]:.1f}%)")
print(f"  • Meilleur mAP@50: {best_map[0]} ({best_map[1]:.1f}%)")

print(f"\n⚠️ CLASSES À AMÉLIORER (< 80%):")
classes_faibles = []
for classe, prec, rec, map_val in zip(classes, precision, rappel, map50):
    if prec < 80 or rec < 80 or map_val < 80:
        classes_faibles.append(classe)
        print(f"  • {classe}: P={prec:.1f}%, R={rec:.1f}%, mAP@50={map_val:.1f}%")

if not classes_faibles:
    print("  Aucune classe sous le seuil de 80%!")

print(f"{'='*80}\n")