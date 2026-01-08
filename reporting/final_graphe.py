# @title
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class GraphiqueGenerator:
    def __init__(self):
        """
        Générateur de graphiques avec données intégrées
        Style unifié et personnalisable facilement
        """
        self.setup_style()
        
        # 🔧 VARIABLES DE PERSONNALISATION - MODIFIEZ ICI SEULEMENT
        self.STYLE_CONFIG = {
            'theme_color': '#2E75B6',          # Couleur principale
            'accent_color': '#ED7D31',         # Couleur d'accent
            'success_color': '#70AD47',        # Couleur succès
            'warning_color': '#FFC000',        # Couleur avertissement
            'danger_color': '#FF0000',         # Couleur danger
            'bg_color': '#FFFFFF',             # Couleur de fond
            'grid_color': '#E6E6E6',           # Couleur grille
            'font_family': 'Arial',            # Police de caractères
            'title_size': 16,                  # Taille titre principal
            'subtitle_size': 14,               # Taille sous-titre
            'label_size': 12,                  # Taille des labels
            'ticks_size': 10,                  # Taille des ticks
            'legend_size': 10,                 # Taille légende
            'fig_width': 12,                   # Largeur figure
            'fig_height': 7,                   # Hauteur figure
            'dpi': 300,                        # Résolution
            'grid_alpha': 0.3,                 # Transparence grille
            'bar_edge_color': 'gray',          # Couleur bordure barres
            'bar_edge_width': 0.5,             # Épaisseur bordure
        }
        
        # 📊 DONNÉES INTÉGRÉES DIRECTEMENT
        self.setup_data()
    
    def setup_style(self):
        """Configure le style global des graphiques"""
        plt.style.use('default')
        sns.set_palette("husl")
        
    def setup_data(self):
        """Intègre toutes les données directement dans la classe"""
        
        # 1. Comparaison mAP@50 et améliorations détaillées
        self.data_comparison = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50_Original": [72, 84, 79, 76],
            "mAP@50_Augmenté": [78.08, 89.08, 85.18, 80.18],
            "Amélioration": [6.08, 5.08, 6.18, 4.18]
        })

        # 2. Métriques détaillées - Données originales
        self.data_original = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50": [72, 84, 79, 76],
            "Précision": [70, 82, 77, 74],
            "Rappel": [68, 80, 74, 70],
            "F1_score": [68.98550725, 80.98765432, 75.47019868, 71.94444444]
        })

        # 3. Métriques détaillées - Données augmentées
        self.data_augmented = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50": [78.08, 89.08, 85.18, 80.18],
            "Précision": [77.5, 88.5, 81.5, 79.5],
            "Rappel": [77.04, 89.04, 82.04, 79.04],
            "F1_score": [76.8, 88.8, 80.8, 78.8]
        })

        # 4. Améliorations détaillées pour toutes les métriques
        self.data_améliorations = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50": [6.08, 5.08, 6.18, 4.18],
            "Précision": [7.5, 6.5, 4.5, 5.5],
            "Rappel": [9.04, 9.04, 8.04, 9.04],
            "F1_score": [7.814492754, 7.812345679, 5.329801325, 6.855555556]
        })
        
        # 4. Performances par classe
        self.data_classes = pd.DataFrame({
          "Classe": [
              "A. flava", "B. tabaci", "Coccinelle", "Degat Jassides", "Dysdercus spp",
              "Earias spp", "Effet phyto", "G. spodoptera", "H. armigera", "Jasside",
              "Larve coccinelle", "Larve syrphe", "P. gossypiella", "Puceron",
              "S. derogata", "S. frugiperda", "Scarabees"
          ],
          "Précision": [
              0.98, 0.964, 0.814, 0.788, 0.978,
              0.981, 0.603, 0.996, 0.986, 0.805,
              0.941, 0.959, 0.925, 0.505,
              0.905, 0.933, 0.995
          ],
          "Rappel": [
              0.987, 0.941, 0.872, 0.682, 0.98,
              0.966, 0.738, 1.000, 0.989, 0.845,
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
        
        # 5. Groupes fonctionnels
        self.data_groups = pd.DataFrame({
          "Groupe": [
              "Chenilles lépidoptères",
              "Piqueurs-suceurs",
              "Prédateurs",
              "Symptômes",
              "Autres"
          ],
          "SSD": [0.712, 0.545, 0.651, 0.788, 0.80],
          "YOLOv8n": [0.973, 0.7197, 0.9523, 0.7185, 0.950],
          "Faster_RCNN_ResNet50": [0.952, 0.685, 0.821, 0.729, 0.901],
          "Faster_RCNN_MobileNetV3": [0.798, 0.652, 0.765, 0.661, 0.84]
      })
        
        # 6. Caractéristiques des modèles
        self.data_characteristics = pd.DataFrame({
          "Modèle": ["SSD MobileNetV3","YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
          "mAP@50_%": [ 77.9,89.08, 82.9, 80.9],
          "Paramètres_M": [ 2.2, 3.0, 41.0, 3.5],
          "GFLOPs": [ 0.6, 8.1, 180.0, 3.2],
          "Complexité": ["Faible", "Très faible", "Très élevée", "Faible"]
      })
        
        # 7. Métriques globales yolo
        self.global_metrics = {
          "F1_Macro": 0.888,        # moyenne simple des 17 F1-scores
          "mAP50_moyen": 0.8908,     # moyenne simple des 17 mAP@50
          "Amélioration_moyenne": 0.0508
        }
    
    def apply_style(self, ax, title, subtitle=None):
        """Applique le style unifié au graphique"""
        # Titre principal
        ax.set_title(
            title, 
            fontsize=self.STYLE_CONFIG['title_size'], 
            fontweight='bold', 
            pad=20,
            color=self.STYLE_CONFIG['theme_color']
        )
        
        # Sous-titre (optionnel)
        if subtitle:
            ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
                   fontsize=self.STYLE_CONFIG['subtitle_size'], 
                   ha='center', va='bottom', style='italic')
        
        # Labels des axes
        ax.set_xlabel(ax.get_xlabel(), fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_ylabel(ax.get_ylabel(), fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        # Ticks
        ax.tick_params(axis='both', which='major', labelsize=self.STYLE_CONFIG['ticks_size'])
        
        # Grille
        ax.grid(True, alpha=self.STYLE_CONFIG['grid_alpha'], linestyle='--', color=self.STYLE_CONFIG['grid_color'])
        
        # Fond
        ax.set_facecolor(self.STYLE_CONFIG['bg_color'])
        
        # Légende (si elle existe)
        legend = ax.get_legend()
        if legend:
            legend.set_title(legend.get_title().get_text(), prop={'size': self.STYLE_CONFIG['legend_size'], 'weight': 'bold'})
            for text in legend.get_texts():
                text.set_fontsize(self.STYLE_CONFIG['legend_size'])
    
    def save_plot(self, fig, filename):
        """Sauvegarde le graphique avec les paramètres de style"""
        fig.savefig(
            filename, 
            dpi=self.STYLE_CONFIG['dpi'], 
            bbox_inches='tight', 
            facecolor=self.STYLE_CONFIG['bg_color']
        )
        plt.close()
        print(f"✅ Graphique sauvegardé: {filename}")

    # ============================
    # NOUVEAUX GRAPHIQUES DEMANDÉS
    # ============================

    def plot_amelioration_boxplot(self):
        """
        Box plot des améliorations mAP@50 des données originales à augmentées
        """
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Préparer les données pour le box plot
        data_original = self.data_comparison["mAP@50_Original"].tolist()
        data_augmente = self.data_comparison["mAP@50_Augmenté"].tolist()
        ameliorations = self.data_comparison["Amélioration"].tolist()
        
        # Créer le box plot
        bp = ax.boxplot([data_original, data_augmente, ameliorations], 
                       labels=['Original', 'Augmenté', 'Amélioration'],
                       patch_artist=True,
                       medianprops={'color': 'black', 'linewidth': 2})
        
        # Personnaliser les couleurs des boîtes
        colors = [self.STYLE_CONFIG['theme_color'], 
                 self.STYLE_CONFIG['success_color'], 
                 self.STYLE_CONFIG['accent_color']]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(self.STYLE_CONFIG['bar_edge_color'])
            patch.set_linewidth(self.STYLE_CONFIG['bar_edge_width'])
        
        # Ajouter les points individuels
        for i, (orig, aug, amel) in enumerate(zip(data_original, data_augmente, ameliorations), 1):
            ax.scatter([1], [orig], color='white', edgecolor='black', s=80, zorder=3)
            ax.scatter([2], [aug], color='white', edgecolor='black', s=80, zorder=3)
            ax.scatter([3], [amel], color='white', edgecolor='black', s=80, zorder=3)
        
        # Configuration
        ax.set_ylabel('mAP@50 (%)', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_xlabel('Type de données', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        # Titre
        title = "Distribution des performances mAP@50\nComparaison Données Originales vs Augmentées"
        subtitle = f"Amélioration moyenne: {np.mean(ameliorations):.2f}% | N={len(data_original)} modèles"
        
        ax.set_title(title, fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
               fontsize=self.STYLE_CONFIG['subtitle_size'], 
               ha='center', va='bottom', style='italic')
        
        # Style
        self.apply_style(ax, "")
        ax.grid(True, alpha=self.STYLE_CONFIG['grid_alpha'], linestyle='--')
        
        # Ajouter des statistiques
        stats_text = f"Original: {np.mean(data_original):.1f}% ± {np.std(data_original):.1f}\n" \
                    f"Augmenté: {np.mean(data_augmente):.1f}% ± {np.std(data_augmente):.1f}\n" \
                    f"Amélioration: +{np.mean(ameliorations):.1f}% ± {np.std(ameliorations):.1f}"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        self.save_plot(fig, "boxplot_amelioration_map50.png")
        return fig

    def plot_comparaison_map50_histogram(self):
        """
        Histogramme de comparaison des mAP@50 par modèle
        Données originales vs augmentées sur le même graphique
        """
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Préparer les données
        modeles = self.data_comparison["Modèle"].tolist()
        x = np.arange(len(modeles))
        width = 0.35  # Largeur des barres
        
        # Créer les barres
        bars1 = ax.bar(x - width/2, self.data_comparison["mAP@50_Original"], 
                      width, label='Données Originales', 
                      color=self.STYLE_CONFIG['theme_color'],
                      edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                      linewidth=self.STYLE_CONFIG['bar_edge_width'],
                      alpha=0.8)
        
        bars2 = ax.bar(x + width/2, self.data_comparison["mAP@50_Augmenté"], 
                      width, label='Données Augmentées', 
                      color=self.STYLE_CONFIG['success_color'],
                      edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                      linewidth=self.STYLE_CONFIG['bar_edge_width'],
                      alpha=0.8)
        
        # Ajouter les valeurs sur les barres
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # Décalage vertical
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, fontweight='bold')
        
        autolabel(bars1)
        autolabel(bars2)
        
        # Ajouter les lignes d'amélioration
        for i, (mod, orig, aug) in enumerate(zip(modeles, 
                                                self.data_comparison["mAP@50_Original"], 
                                                self.data_comparison["mAP@50_Augmenté"])):
            amel = self.data_comparison["Amélioration"].iloc[i]
            ax.plot([i - width/2, i + width/2], [orig, aug], 
                   color=self.STYLE_CONFIG['accent_color'], 
                   linewidth=2, marker='o', markersize=6)
            
            # Annotation d'amélioration
            ax.text(i, max(orig, aug) + 2, f"+{amel:.1f}%", 
                   ha='center', va='bottom', 
                   fontsize=9, fontweight='bold',
                   color=self.STYLE_CONFIG['accent_color'])
        
        # Configuration
        ax.set_xlabel('Modèles', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_ylabel('mAP@50 (%)', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modeles, rotation=0, fontsize=self.STYLE_CONFIG['ticks_size'])
        ax.set_ylim(60, 95)  # Échelle fixe pour meilleure comparaison
        
        # Titre
        title = "Comparaison des performances mAP@50 par modèle\nDonnées Originales vs Augmentées"
        ax.set_title(title, fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        
        # Légende EN DEHORS du cadre
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                 borderaxespad=0., fontsize=self.STYLE_CONFIG['legend_size'])
        
        # Style
        self.apply_style(ax, "")
        ax.grid(True, axis='y', alpha=self.STYLE_CONFIG['grid_alpha'], linestyle='--')
        
        # Ajouter des statistiques globales
        moy_orig = np.mean(self.data_comparison["mAP@50_Original"])
        moy_aug = np.mean(self.data_comparison["mAP@50_Augmenté"])
        moy_amel = np.mean(self.data_comparison["Amélioration"])
        
        stats_text = f"mAP@50 moyen Original: {moy_orig:.1f}%\n" \
                    f"mAP@50 moyen Augmenté: {moy_aug:.1f}%\n" \
                    f"Amélioration moyenne: +{moy_amel:.1f}%"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajuster pour la légende externe
        self.save_plot(fig, "comparaison_map50_histogram.png")
        return fig

    def plot_graphe_complet_amelioration(self):
        """
        Graphique complet montrant l'amélioration avec toutes les métriques
        """
        fig, axes = plt.subplots(2, 2, figsize=(self.STYLE_CONFIG['fig_width']*1.2, 
                                               self.STYLE_CONFIG['fig_height']*1.5))
        axes = axes.flatten()
        
        # 1. Graphique 1: Comparaison mAP@50 (barres groupées)
        ax1 = axes[0]
        modeles = self.data_comparison["Modèle"].tolist()
        x = np.arange(len(modeles))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, self.data_comparison["mAP@50_Original"], 
                       width, label='Original', 
                       color=self.STYLE_CONFIG['theme_color'], alpha=0.7)
        bars2 = ax1.bar(x + width/2, self.data_comparison["mAP@50_Augmenté"], 
                       width, label='Augmenté', 
                       color=self.STYLE_CONFIG['success_color'], alpha=0.7)
        
        ax1.set_xlabel('Modèles')
        ax1.set_ylabel('mAP@50 (%)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(modeles, rotation=15)
        ax1.set_title('Performance mAP@50 par modèle', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Graphique 2: Améliorations (barres horizontales)
        ax2 = axes[1]
        y_pos = np.arange(len(modeles))
        ameliorations = self.data_améliorations[["mAP@50", "Précision", "Rappel"]].mean(axis=1)
        
        bars = ax2.barh(y_pos, self.data_comparison["Amélioration"], 
                       color=self.data_comparison["Amélioration"].apply(
                           lambda x: self.STYLE_CONFIG['success_color'] if x > 5 
                           else self.STYLE_CONFIG['warning_color']),
                       alpha=0.8)
        
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(modeles)
        ax2.set_xlabel('Amélioration (%)')
        ax2.set_title('Amélioration mAP@50 après augmentation', fontsize=12, fontweight='bold')
        
        # Ajouter les valeurs
        for bar in bars:
            width = bar.get_width()
            ax2.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{width:.1f}%', ha='left', va='center', fontsize=9)
        
        # 3. Graphique 3: Diagramme en radar des améliorations
        ax3 = axes[2]
        metrics = ['mAP@50', 'Précision', 'Rappel', 'F1_score']
        n_metrics = len(metrics)
        
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        for idx, modele in enumerate(modeles):
            values = []
            for metric in metrics:
                if metric == 'mAP@50':
                    values.append(self.data_améliorations.loc[idx, metric])
                else:
                    values.append(self.data_améliorations.loc[idx, metric])
            values += values[:1]
            
            ax3.plot(angles, values, 'o-', linewidth=2, label=modele)
            ax3.fill(angles, values, alpha=0.1)
        
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metrics)
        ax3.set_title('Amélioration par métrique', fontsize=12, fontweight='bold')
        ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
        ax3.grid(True)
        
        # 4. Graphique 4: Heatmap des améliorations
        ax4 = axes[3]
        heatmap_data = self.data_améliorations.set_index('Modèle')[['mAP@50', 'Précision', 'Rappel']]
        
        im = ax4.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
        
        ax4.set_xticks(np.arange(len(heatmap_data.columns)))
        ax4.set_yticks(np.arange(len(heatmap_data.index)))
        ax4.set_xticklabels(heatmap_data.columns)
        ax4.set_yticklabels(heatmap_data.index)
        
        plt.setp(ax4.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Ajouter les valeurs dans les cases
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                text = ax4.text(j, i, f'{heatmap_data.iloc[i, j]:.1f}',
                               ha="center", va="center", color="black", fontsize=9)
        
        ax4.set_title('Heatmap des améliorations (%)', fontsize=12, fontweight='bold')
        
        # Titre général
        fig.suptitle('Analyse complète des améliorations après augmentation des données', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        self.save_plot(fig, "graphe_complet_amelioration.png")
        return fig

    def plot_graphe_principal(self):
        """
        Graphique principal combinant box plot et histogramme
        """
        fig = plt.figure(figsize=(self.STYLE_CONFIG['fig_width']*1.5, 
                                 self.STYLE_CONFIG['fig_height']))
        
        # Grille de 1x2
        gs = fig.add_gridspec(1, 2, wspace=0.3)
        
        # 1. Box plot à gauche
        ax1 = fig.add_subplot(gs[0, 0])
        
        data_original = self.data_comparison["mAP@50_Original"].tolist()
        data_augmente = self.data_comparison["mAP@50_Augmenté"].tolist()
        ameliorations = self.data_comparison["Amélioration"].tolist()
        
        bp = ax1.boxplot([data_original, data_augmente, ameliorations], 
                        labels=['Original', 'Augmenté', 'Δ Amélioration'],
                        patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2})
        
        colors = [self.STYLE_CONFIG['theme_color'], 
                 self.STYLE_CONFIG['success_color'], 
                 self.STYLE_CONFIG['accent_color']]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('mAP@50 (%)', fontweight='bold')
        ax1.set_title('Distribution des performances', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        # 2. Histogramme à droite
        ax2 = fig.add_subplot(gs[0, 1])
        
        modeles = self.data_comparison["Modèle"].tolist()
        x = np.arange(len(modeles))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, self.data_comparison["mAP@50_Original"], 
                       width, label='Original', 
                       color=self.STYLE_CONFIG['theme_color'], alpha=0.7)
        bars2 = ax2.bar(x + width/2, self.data_comparison["mAP@50_Augmenté"], 
                       width, label='Augmenté', 
                       color=self.STYLE_CONFIG['success_color'], alpha=0.7)
        
        ax2.set_xlabel('Modèles', fontweight='bold')
        ax2.set_ylabel('mAP@50 (%)', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(modeles, rotation=15)
        ax2.set_title('Comparaison par modèle', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        ax2.grid(True, alpha=0.3, linestyle='--')
        
        # Ajouter les valeurs d'amélioration
        for i, (orig, aug) in enumerate(zip(self.data_comparison["mAP@50_Original"], 
                                           self.data_comparison["mAP@50_Augmenté"])):
            amel = self.data_comparison["Amélioration"].iloc[i]
            ax2.text(i, max(orig, aug) + 1, f"+{amel:.1f}%", 
                    ha='center', va='bottom', 
                    fontsize=8, fontweight='bold',
                    color=self.STYLE_CONFIG['accent_color'])
        
        # Titre général
        fig.suptitle('Impact de l\'augmentation des données sur les performances mAP@50', 
                    fontsize=14, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        self.save_plot(fig, "graphe_principal_comparaison.png")
        return fig

    def plot_boxplot_ecart_amelioration_simple(self):
        """
        Box plot simple des écarts d'amélioration
        """
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Récupérer les données
        ameliorations = self.data_comparison["Amélioration"]
        
        # Créer le box plot
        ax.boxplot(ameliorations, 
                vert=True,
                patch_artist=True,
                boxprops=dict(facecolor=self.STYLE_CONFIG['accent_color'], alpha=0.7),
                medianprops=dict(color='black', linewidth=2),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'),
                flierprops=dict(marker='o', color='red', alpha=0.5))
        
        # Ajouter les points
        for i, val in enumerate(ameliorations, 1):
            ax.scatter([1], [val], color='white', edgecolor='black', s=80, zorder=3)
        
        # Configuration
        ax.set_ylabel('Amélioration mAP@50 (%)', fontweight='bold')
        ax.set_xticklabels(['Amélioration'])
        ax.set_title('Écarts d\'amélioration des performances', fontweight='bold')
        
        # Ajouter les valeurs
        for i, (modele, val) in enumerate(zip(self.data_comparison["Modèle"], ameliorations)):
            ax.text(1.1, val, f'{modele}: +{val:.2f}%', va='center', fontsize=9)
        
        self.apply_style(ax, "Distribution des améliorations mAP@50")
        
        plt.tight_layout()
        self.save_plot(fig, "boxplot_ecarts_simple.png")
        return fig
    
    def plot_boxplot_ecart_amelioration(self):
        """
        Box plot des écarts d'amélioration mAP@50 entre les modèles
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Données d'amélioration
        ameliorations = self.data_comparison["Amélioration"].tolist()
        modeles = self.data_comparison["Modèle"].tolist()
        
        # Créer le box plot
        bp = ax.boxplot(ameliorations, 
                    patch_artist=True,
                    labels=['Écarts d\'amélioration'],
                    showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'red'},
                    medianprops={'color': 'black', 'linewidth': 2})
        
        # Personnaliser la boîte
        bp['boxes'][0].set_facecolor(self.STYLE_CONFIG['accent_color'])
        bp['boxes'][0].set_alpha(0.7)
        bp['boxes'][0].set_edgecolor(self.STYLE_CONFIG['bar_edge_color'])
        bp['boxes'][0].set_linewidth(self.STYLE_CONFIG['bar_edge_width'])
        
        # Ajouter les points individuels avec labels
        for i, (amel, modele) in enumerate(zip(ameliorations, modeles), 1):
            ax.scatter([1], [amel], color='white', edgecolor='black', s=100, zorder=3)
            ax.text(1.05, amel, f'{modele}: +{amel:.2f}%', 
                va='center', fontsize=9, fontweight='bold')
        
        # Calculer les statistiques
        mean_val = np.mean(ameliorations)
        median_val = np.median(ameliorations)
        std_val = np.std(ameliorations)
        min_val = min(ameliorations)
        max_val = max(ameliorations)
        
        # Configuration
        ax.set_ylabel('Amélioration mAP@50 (%)', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_title('Distribution des écarts d\'amélioration après augmentation des données', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        
        # Sous-titre avec statistiques
        subtitle = f"Moyenne: {mean_val:.2f}% | Médiane: {median_val:.2f}% | Écart-type: {std_val:.2f}%"
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
            fontsize=self.STYLE_CONFIG['subtitle_size'], 
            ha='center', va='bottom', style='italic')
        
        # Ajouter des lignes de référence
        ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.5, label=f'Moyenne: {mean_val:.2f}%')
        ax.axhline(y=5.0, color='green', linestyle=':', alpha=0.5, label='Seuil: 5%')
        
        # Style
        self.apply_style(ax, "")
        ax.grid(True, alpha=self.STYLE_CONFIG['grid_alpha'], linestyle='--')
        
        # Ajouter la légende
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=self.STYLE_CONFIG['legend_size'])
        
        # Zone de statistiques détaillées
        stats_text = f"Minimum: {min_val:.2f}%\n" \
                    f"Maximum: {max_val:.2f}%\n" \
                    f"Écart: {max_val - min_val:.2f}%\n" \
                    f"Nombre: {len(ameliorations)} modèles"
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Ajuster les limites
        buffer = 0.5
        ax.set_ylim(min_val - buffer, max_val + buffer + 1)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Ajuster pour la légende externe
        self.save_plot(fig, "boxplot_ecart_amelioration.png")
        return fig
    
    def plot_boxplot_classes_yolo(self):
        """
        Box plot des métriques par classe pour le modèle YOLO
        Inclut Précision, Rappel, F1-score avec ligne du F1 moyen
        """
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Préparer les données - convertir en pourcentage
        precision = (self.data_classes["Précision"] * 100).tolist()
        rappel = (self.data_classes["Rappel"] * 100).tolist()
        f1_score = (self.data_classes["F1_score"] * 100).tolist()
        
        # Calculer le F1-score moyen (macro)
        f1_moyen = np.mean(f1_score)
        
        # Créer le box plot
        data_to_plot = [precision, rappel, f1_score]
        labels = ['Précision (P)', 'Rappel (R)', 'F1-score']
        
        bp = ax.boxplot(data_to_plot, 
                    labels=labels,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={'marker': 'D', 'markerfacecolor': 'white', 'markeredgecolor': 'red', 'markersize': 8},
                    medianprops={'color': 'black', 'linewidth': 2.5})
        
        # Personnaliser les couleurs des boîtes
        colors = [self.STYLE_CONFIG['theme_color'], 
                self.STYLE_CONFIG['success_color'], 
                self.STYLE_CONFIG['accent_color']]
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor(self.STYLE_CONFIG['bar_edge_color'])
            patch.set_linewidth(self.STYLE_CONFIG['bar_edge_width'])
        
        # Ajouter la ligne horizontale du F1 moyen
        ax.axhline(y=f1_moyen, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'F1 moyen: {f1_moyen:.1f}%')
        
        # Ajouter une zone ombrée pour le F1 moyen ± écart-type
        f1_std = np.std(f1_score)
        ax.fill_between([0.5, 3.5], f1_moyen - f1_std, f1_moyen + f1_std, 
                    alpha=0.2, color='red', label=f'±1 σ ({f1_std:.1f}%)')
        
        # Configuration
        ax.set_ylabel('Score (%)', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_xlabel('Métriques', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        # Titre
        title = "Distribution des métriques par classe - Modèle YOLOv8n"
        subtitle = f"Basé sur {len(precision)} classes | F1-score macro: {f1_moyen:.2f}%"
        
        ax.set_title(title, fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.text(0.5, 0.95, subtitle, transform=ax.transAxes, 
            fontsize=self.STYLE_CONFIG['subtitle_size'], 
            ha='center', va='bottom', style='italic')
        
        # Style
        self.apply_style(ax, "")
        ax.grid(True, alpha=self.STYLE_CONFIG['grid_alpha'], linestyle='--', axis='y')
        
        # Ajouter la légende EN DEHORS du cadre
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), 
                fontsize=self.STYLE_CONFIG['legend_size'],
                framealpha=0.9)
        
        # Calculer et afficher les statistiques
        stats_text = []
        for i, (label, data) in enumerate(zip(labels, data_to_plot)):
            stats_text.append(f"{label}:")
            stats_text.append(f"  Moyenne: {np.mean(data):.1f}%")
            stats_text.append(f"  Médiane: {np.median(data):.1f}%")
            stats_text.append(f"  Min: {np.min(data):.1f}% | Max: {np.max(data):.1f}%")
            stats_text.append(f"  σ: {np.std(data):.1f}%")
            if i < len(labels) - 1:
                stats_text.append("")
        
        stats_str = "\n".join(stats_text)
        ax.text(1.02, 0.3, stats_str, transform=ax.transAxes, 
            fontsize=9, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Ajuster les limites
        all_data = precision + rappel + f1_score
        y_min = max(0, np.min(all_data) - 5)
        y_max = min(100, np.max(all_data) + 5)
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout(rect=[0, 0, 0.78, 1])  # Ajuster pour la légende externe
        self.save_plot(fig, "boxplot_classes_yolo.png")
        return fig

    def plot_boxplot_classes_yolo_detaille(self):
        """
        Box plot détaillé avec points individuels pour chaque classe
        """
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width']*1.2, self.STYLE_CONFIG['fig_height']))
        
        # Préparer les données
        classes = self.data_classes["Classe"].tolist()
        precision = (self.data_classes["Précision"] * 100).tolist()
        rappel = (self.data_classes["Rappel"] * 100).tolist()
        f1_score = (self.data_classes["F1_score"] * 100).tolist()
        f1_moyen = np.mean(f1_score)
        
        # Créer les box plots
        data_to_plot = [precision, rappel, f1_score]
        labels = ['Précision', 'Rappel', 'F1-score']
        
        positions = np.arange(1, len(labels) + 1)
        bp = ax.boxplot(data_to_plot, positions=positions,
                    widths=0.6,
                    patch_artist=True,
                    showfliers=False)  # On ajoutera les points manuellement
        
        # Couleurs
        colors = ['#4C72B0', '#55A868', '#C44E52']  # Bleu, Vert, Rouge
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Ajouter les points individuels pour chaque classe
        for i, (p, r, f, classe) in enumerate(zip(precision, rappel, f1_score, classes)):
            # Jitter pour éviter la superposition
            jitter = np.random.normal(0, 0.02, 1)[0]
            
            ax.scatter(1 + jitter, p, color=colors[0], alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
            ax.scatter(2 + jitter, r, color=colors[1], alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
            ax.scatter(3 + jitter, f, color=colors[2], alpha=0.6, s=40, edgecolor='white', linewidth=0.5)
            
            # Connecter les points de la même classe
            ax.plot([1 + jitter, 2 + jitter, 3 + jitter], [p, r, f], 
                color='gray', alpha=0.2, linewidth=0.5)
        
        # Ligne du F1 moyen
        ax.axhline(y=f1_moyen, color='red', linestyle='--', linewidth=2, 
                label=f'F1 moyen: {f1_moyen:.1f}%')
        
        # Configuration
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=self.STYLE_CONFIG['label_size'])
        ax.set_ylabel('Score (%)', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        # Titre
        ax.set_title('Distribution des métriques par classe - YOLOv8n', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        
        # Légende
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
        
        # Style
        self.apply_style(ax, "")
        ax.grid(True, alpha=0.3, axis='y')
        
        # Informations sur le F1
        ax.text(0.02, 0.98, f'F1-score moyen: {f1_moyen:.2f}%\nÉcart-type: {np.std(f1_score):.2f}%\nn={len(classes)} classes', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        self.save_plot(fig, "boxplot_classes_yolo_detaille.png")
        return fig

    def generate_all_plots(self):
        """Génère tous les graphiques"""
        print("🎨 Génération des graphiques...")
        
        plots = []
        
        # 1. Box plot des améliorations
        print("📦 1. Box plot des améliorations...")
        fig1 = self.plot_amelioration_boxplot()
        plots.append(("boxplot_amelioration", fig1))
        
        # 2. Histogramme de comparaison
        print("📊 2. Histogramme de comparaison...")
        fig2 = self.plot_comparaison_map50_histogram()
        plots.append(("histogram_comparaison", fig2))
        
        # 3. Graphique complet
        print("📈 3. Graphique complet d'amélioration...")
        fig3 = self.plot_graphe_complet_amelioration()
        plots.append(("graphe_complet", fig3))
        
        # 4. Graphique principal
        print("🏆 4. Graphique principal...")
        fig4 = self.plot_graphe_principal()
        plots.append(("graphe_principal", fig4))

        print("🏆 5. Boxplot écart amélioration simple...")
        fig5 = self.plot_boxplot_ecart_amelioration_simple()
        plots.append(("boxplot_ecart_amelioration_simple", fig5))
        
        print("🏆 6. Boxplot écart amélioration détaillé...")
        fig6 = self.plot_boxplot_ecart_amelioration()
        plots.append(("boxplot_ecart_amelioration", fig6))

        print("🏆 7. Boxplot classes YOLO...")
        fig7 = self.plot_boxplot_classes_yolo()
        plots.append(("boxplot_classes_yolo", fig7))

        print("🏆 8. Boxplot classes YOLO détaillé...")
        fig8 = self.plot_boxplot_classes_yolo_detaille()
        plots.append(("boxplot_classes_yolo_detaille", fig8))

        print("\n✅ Tous les graphiques ont été générés et sauvegardés !")
        print("📁 Fichiers créés:")
        print("   - boxplot_amelioration_map50.png")
        print("   - comparaison_map50_histogram.png")
        print("   - graphe_complet_amelioration.png")
        print("   - graphe_principal_comparaison.png")
        
        return plots


# ============================================
# UTILISATION
# ============================================

# Créer l'instance et générer tous les graphiques
if __name__ == "__main__":
    generator = GraphiqueGenerator()
    generator.generate_all_plots()