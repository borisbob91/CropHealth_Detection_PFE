import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Données
data_groups = pd.DataFrame({
    "Groupe": [
        "Chenilles\nlépidoptères",
        "Piqueurs-\nsuceurs",
        "Prédateurs",
        "Symptômes",
        "Autres"
    ],
    "SSD": [0.712, 0.545, 0.651, 0.688, 0.80],
    "YOLOv8n": [0.973, 0.7197, 0.9523, 0.7185, 0.950],
    "Faster_RCNN_ResNet50": [0.952, 0.685, 0.821, 0.729, 0.901],
    "Faster_RCNN_MobileNetV3": [0.798, 0.652, 0.765, 0.661, 0.84]
})

# Préparation des données
# Noms d'affichage pour la légende
noms_modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
# Noms des colonnes dans le DataFrame
colonnes_modeles = ['SSD', 'YOLOv8n', 'Faster_RCNN_ResNet50', 'Faster_RCNN_MobileNetV3']

groupes = data_groups["Groupe"].tolist()

# Configuration du graphique
plt.figure(figsize=(14, 10))

x = np.arange(len(groupes))
largeur = 0.2  # Un peu plus étroit car 4 barres par groupe

# Couleurs cohérentes avec votre style (adaptées pour 4 modèles)
couleurs = [
    '#2E86AB',  # Bleu pour SSD
    '#A23B72',  # Rose pour YOLOv8n
    '#2E8B57',  # Vert pour Faster R-CNN ResNet50
    '#FF8C00'   # Orange pour Faster R-CNN MobileNetV3
]

# Création des barres groupées
bars = []
for i, (modele_aff, modele_col) in enumerate(zip(noms_modeles, colonnes_modeles)):
    valeurs = data_groups[modele_col].values * 100  # Conversion en pourcentage
    bar = plt.bar(x + i*largeur - (len(noms_modeles)-1)*largeur/2, 
                  valeurs, largeur, 
                  label=modele_aff, 
                  color=couleurs[i], 
                  edgecolor='black', linewidth=2, alpha=0.9)
    bars.append(bar)

# Personnalisation
#plt.title('Performance mAP@50 par Groupe Fonctionnel', 
#          fontsize=22, fontweight='bold', pad=30)
plt.ylabel('mAP@50 (%)', fontsize=16, fontweight='bold')
plt.xlabel('Groupes Fonctionnels', fontsize=16, fontweight='bold')
plt.xticks(x, groupes, fontsize=16)
plt.ylim(40, 105)  # Ajusté pour les données
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres
for i, bar_set in enumerate(bars):
    for bar in bar_set:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height - 3,
                f'{height:.1f}', ha='center', va='top',
                fontsize=14, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=couleurs[i], 
                         alpha=0.9))

# Légende en haut, centrée
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=4, fontsize=14, framealpha=0.9, frameon=True)

# Ajustements finaux
plt.tight_layout()
plt.subplots_adjust(top=0.85)  # Espace pour la légende

# Annotations pour les meilleurs scores par groupe
for idx, groupe in enumerate(groupes):
    scores = [data_groups[col].iloc[idx] * 100 for col in colonnes_modeles]
    best_idx = np.argmax(scores)
    best_x = x[idx] + best_idx*largeur - (len(noms_modeles)-1)*largeur/2
    
    plt.annotate('Meilleur', 
                 xy=(best_x, scores[best_idx]),
                 xytext=(best_x, scores[best_idx] + 5),
                 ha='center', va='bottom',
                 fontsize=12, fontweight='bold',
                 color='green', arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

plt.show()