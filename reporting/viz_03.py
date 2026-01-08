import matplotlib.pyplot as plt
import numpy as np

# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']

# Données augmentées
f1_augmente = [76.8, 88.8, 80.8, 78.8]
map_augmente = [78.08, 89.08, 85.18, 80.18]

# Configuration du graphique
plt.figure(figsize=(12, 8))

x = np.arange(len(modeles))
largeur = 0.35

# Couleurs
couleur_f1 = '#2E86AB'  # Bleu pour F1
couleur_map = '#A23B72'  # Rose pour mAP

# Création des barres groupées
bars_f1 = plt.bar(x - largeur/2, f1_augmente, largeur, 
                  label='F1-score', color=couleur_f1, 
                  edgecolor='black', linewidth=2, alpha=0.9)

bars_map = plt.bar(x + largeur/2, map_augmente, largeur, 
                   label='mAP@50', color=couleur_map, 
                   edgecolor='black', linewidth=2, alpha=0.9)

# Personnalisation
plt.title('Comparaison F1-score et mAP@50 - Données Augmentées', 
          fontsize=20, fontweight='bold', pad=25)
plt.ylabel('Score (%)', fontsize=18, fontweight='bold')
plt.xlabel('Modèles', fontsize=18, fontweight='bold')
plt.xticks(x, modeles, fontsize=16)
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# Valeurs sur les barres - F1-score
for bar, valeur in zip(bars_f1, f1_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'{valeur:.1f}', ha='center', va='top',
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a5276', alpha=0.9))

# Valeurs sur les barres - mAP@50
for bar, valeur in zip(bars_map, map_augmente):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 2,
            f'{valeur:.1f}', ha='center', va='top',
            fontsize=14, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#7d3c98', alpha=0.9))

# Légende en haut, centrée
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.03), 
           ncol=2, fontsize=16, framealpha=0.9, frameon=True)

# Ajustements finaux
plt.tight_layout()
plt.subplots_adjust(top=0.88)  # Espace pour la légende
plt.show()



# Données
modeles = ['SSD', 'YOLOv8n', 'Faster R-CNN\nResNet50', 'Faster R-CNN\nMobileNetV3']
f1_augmente = [76.8, 88.8, 80.8, 78.8]
map_augmente = [78.08, 89.08, 85.18, 80.18]

# Configuration
plt.figure(figsize=(12, 8))

x = np.arange(len(modeles))
largeur = 0.35

# Couleurs
couleur_f1 = '#3498db'  # Bleu clair
couleur_map = '#e74c3c'  # Rouge

# Barres
bars_f1 = plt.bar(x - largeur/2, f1_augmente, largeur, 
                  label=f'F1-score', color=couleur_f1, 
                  edgecolor='black', linewidth=1.5, )

bars_map = plt.bar(x + largeur/2, map_augmente, largeur, 
                   label=f'mAP@50', color=couleur_map, 
                   edgecolor='black', linewidth=1.5)

# Personnalisation
#plt.title('Performance des Modèles - Données Augmentées', 
#          fontsize=20, fontweight='bold', pad=25)
plt.ylabel('Score (%)', fontsize=18, fontweight='bold')
plt.xlabel('Modèles', fontsize=18, fontweight='bold')
plt.xticks(x, modeles, fontsize=16)
plt.ylim(70, 95)
plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# Valeurs au-dessus des barres (plus visible)
def ajouter_valeurs(bars, decalage=0.5):
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + decalage,
                f'{height:.1f}', ha='center', va='bottom',
                fontsize=13, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', 
                         facecolor='white', 
                         edgecolor='gray', alpha=0.9))

ajouter_valeurs(bars_f1, 0.3)
ajouter_valeurs(bars_map, 0.3)

# Légende
plt.legend(loc='upper left', bbox_to_anchor=(0.6, 1.03), 
           ncol=3, fontsize=16, framealpha=1, frameon=True,
           edgecolor='black')

# Annotations pour le meilleur score
best_f1_idx = np.argmax(f1_augmente)
best_map_idx = np.argmax(map_augmente)

if best_f1_idx == best_map_idx:
    plt.annotate('Meilleur modèle', 
                 xy=(best_f1_idx, max(f1_augmente[best_f1_idx], map_augmente[best_map_idx])),
                 xytext=(best_f1_idx, 94),
                 ha='center', va='bottom',
                 fontsize=14, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))

plt.tight_layout()
plt.show()