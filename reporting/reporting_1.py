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
        
        # 5. Groupes fonctionnels
        self.data_groups = pd.DataFrame({
          "Groupe": [
              "Chenilles lépidoptères",
              "Piqueurs-suceurs",
              "Prédateurs",
              "Symptômes",
              "Autres"
          ],
          "SSD": [0.712, 0.545, 0.651, 0.688, 0.80],
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
        
        # Légende
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
        print(f"✅ {filename}")
    
    def plot_comparison_map50(self):
        """Graphique 1: Comparaison mAP@50 original vs augmenté"""
        df = self.data_comparison
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Configuration
        x = np.arange(len(df['Modèle']))
        width = 0.35
        
        # Barres original vs augmenté
        bars_original = ax.bar(x - width/2, df['mAP@50_Original'], width, 
                              label='Original', 
                              color=self.STYLE_CONFIG['theme_color'],
                              edgecolor=self.STYLE_CONFIG['bar_edge_color'], 
                              linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        bars_augmente = ax.bar(x + width/2, df['mAP@50_Augmenté'], width, 
                              label='Augmenté', 
                              color=self.STYLE_CONFIG['accent_color'],
                              edgecolor=self.STYLE_CONFIG['bar_edge_color'], 
                              linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        # Ajout des valeurs
        for bars in [bars_original, bars_augmente]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 0.005, 
                       f'{height:.3f}', ha='center', va='bottom', 
                       fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Modèles')
        ax.set_ylabel('mAP@50')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Modèle'])
        ax.legend()
        ax.set_ylim(0, 1.0)
        
        self.apply_style(ax, "Comparaison des performances mAP@50", "Données originales vs données augmentées")
        self.save_plot(fig, "01_comparaison_map50.png")
    
    def plot_metrics_original(self):
        """Graphique 2a: Métriques détaillées - Données originales"""
        df = self.data_original
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        x = np.arange(len(df['Modèle']))
        bar_width = 0.2
        metrics = ['mAP@50', 'Précision', 'Rappel', 'F1_score']
        
        # Palette de couleurs
        colors = [
            self.STYLE_CONFIG['theme_color'],
            self.STYLE_CONFIG['accent_color'], 
            self.STYLE_CONFIG['success_color'],
            self.STYLE_CONFIG['warning_color']
        ]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(x + i * bar_width, df[metric], bar_width, 
                         label=metric.replace('_', ' '), 
                         color=color,
                         edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                         linewidth=self.STYLE_CONFIG['bar_edge_width'])
            
            # Valeurs sur les barres
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Modèles')
        ax.set_ylabel('Score (%)')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(df['Modèle'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)
        
        self.apply_style(ax, "Métriques détaillées par modèle", "Données originales")
        self.save_plot(fig, "02a_metriques_originales.png")
    
    def plot_metrics_augmented(self):
        """Graphique 2b: Métriques détaillées - Données augmentées"""
        df = self.data_augmented
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        x = np.arange(len(df['Modèle']))
        bar_width = 0.2
        metrics = ['mAP@50', 'Précision', 'Rappel', 'F1_score']
        
        colors = [
            self.STYLE_CONFIG['theme_color'],
            self.STYLE_CONFIG['accent_color'], 
            self.STYLE_CONFIG['success_color'],
            self.STYLE_CONFIG['warning_color']
        ]
        
        for i, (metric, color) in enumerate(zip(metrics, colors)):
            bars = ax.bar(x + i * bar_width, df[metric], bar_width, 
                         label=metric.replace('_', ' '), 
                         color=color,
                         edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                         linewidth=self.STYLE_CONFIG['bar_edge_width'])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                       f'{height:.1f}', ha='center', va='bottom', 
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Modèles')
        ax.set_ylabel('Score (%)')
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels(df['Modèle'])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 100)
        
        self.apply_style(ax, "Métriques détaillées par modèle", "Données augmentées")
        self.save_plot(fig, "02b_metriques_augmentees.png")
    
    def plot_class_performance(self):
        """Graphique 3: Performance par classe"""
        df = self.data_classes
        
        fig, ax = plt.subplots(figsize=(16, self.STYLE_CONFIG['fig_height']))
        
        # Préparation des données
        df_melted = df.melt(id_vars='Classe', 
                           value_vars=['Précision', 'Rappel', 'F1_score'], 
                           var_name='Métrique', 
                           value_name='Score')
        
        # Couleurs personnalisées
        colors = {
            'Précision': self.STYLE_CONFIG['theme_color'], 
            'Rappel': self.STYLE_CONFIG['accent_color'], 
            'F1_score': self.STYLE_CONFIG['success_color']
        }
        
        # Graphique à barres groupées
        sns.barplot(data=df_melted, x='Classe', y='Score', hue='Métrique', 
                   palette=colors, ax=ax, 
                   edgecolor=self.STYLE_CONFIG['bar_edge_color'])
        
        # Ajout des valeurs F1
        for i, (classe, f1) in enumerate(zip(df['Classe'], df['F1_score'])):
            ax.text(i, f1 + 0.02, f'{f1:.3f}', ha='center', va='bottom', 
                   fontsize=8, fontweight='bold', color='darkred')
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('Score')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_ylim(0, 1.1)
        
        self.apply_style(ax, "Performance par classe (Yolo Données augmentées)")
        self.save_plot(fig, "03_performance_classes.png")
    
    def plot_f1_by_class(self):
        """Graphique 4: F1-score par classe avec seuils colorés"""
        df = self.data_classes
        
        fig, ax = plt.subplots(figsize=(16, self.STYLE_CONFIG['fig_height']))
        
        # Couleurs conditionnelles selon le F1-score
        colors = []
        for f1 in df['F1_score']:
            if f1 >= 0.80:
                colors.append(self.STYLE_CONFIG['success_color'])  # Vert
            elif f1 >= 0.50:
                colors.append(self.STYLE_CONFIG['warning_color'])  # Orange
            else:
                colors.append(self.STYLE_CONFIG['danger_color'])   # Rouge
        
        # Barres avec couleurs conditionnelles
        bars = ax.bar(df['Classe'], df['F1_score'], 
                     color=colors,
                     edgecolor=self.STYLE_CONFIG['bar_edge_color'],
                     linewidth=self.STYLE_CONFIG['bar_edge_width'])
        
        # Valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.015, 
                   f'{height:.3f}', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
        
        # Seuils de référence
        ax.axhline(self.global_metrics['F1_Macro'], linestyle='--', 
                  color='red', linewidth=2, 
                  label=f"F1 Macro: {self.global_metrics['F1_Macro']}")
        
        ax.set_xlabel('Classes')
        ax.set_ylabel('F1-score')
        ax.set_xticklabels(df['Classe'], rotation=45, ha='right')
        ax.set_ylim(0, 1.1)
        ax.legend()
        
        self.apply_style(ax, "F1-score par classe avec seuils de performance", "Vert≥0.8, Orange≥0.5, Rouge<0.5")
        self.save_plot(fig, "04_f1_score_classes.png")
    
    def plot_functional_groups(self):
        """Graphique 5: Groupes fonctionnels - Comparaison des modèles"""
        df = self.data_groups
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        # Préparation des données
        df_melted = df.melt(id_vars='Groupe', 
                           var_name='Modèle', 
                           value_name='mAP@50')
        
        # Palette de couleurs pour les modèles
        colors = {
            'SSD': self.STYLE_CONFIG['theme_color'],
            'YOLOv8n': self.STYLE_CONFIG['accent_color'],
            'Faster_RCNN_ResNet50': self.STYLE_CONFIG['success_color'],
            'Faster_RCNN_MobileNetV3': self.STYLE_CONFIG['warning_color']
        }
        
        # Graphique à barres groupées
        sns.barplot(data=df_melted, x='Groupe', y='mAP@50', hue='Modèle', 
                   palette=colors, ax=ax, 
                   edgecolor=self.STYLE_CONFIG['bar_edge_color'])
        
        # Ajout des valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', 
                        fontsize=8, fontweight='bold', padding=3)
        
        ax.set_xlabel('Groupes fonctionnels')
        ax.set_ylabel('mAP@50')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        ax.set_ylim(0, 1.05)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        self.apply_style(ax, "Performance par groupe fonctionnel")
        self.save_plot(fig, "05_groupes_fonctionnels.png")
    
    def plot_model_characteristics(self):
        """Graphique 6: Caractéristiques et complexité des modèles"""
        df = self.data_characteristics
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, self.STYLE_CONFIG['fig_height']))
        
        # Graphique 1: Scatter plot Performance vs Complexité
        scatter = ax1.scatter(df['Paramètres_M'], df['mAP@50_%'], 
                            s=df['GFLOPs']*5,  # Taille proportionnelle aux GFLOPs
                            alpha=0.7, 
                            c=range(len(df)), 
                            cmap='viridis',
                            edgecolors='black', linewidth=0.5)
        
        # Annotation des points
        for i, row in df.iterrows():
            ax1.annotate(row['Modèle'], 
                        (row['Paramètres_M'], row['mAP@50_%']),
                        xytext=(8, 8), textcoords='offset points', 
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
        
        ax1.set_xlabel('Paramètres (Millions)')
        ax1.set_ylabel('mAP@50 (%)')
        ax1.set_title('Performance vs Complexité', fontweight='bold')
        
        # Graphique 2: Barres comparatives mAP@50 et GFLOPs
        x = np.arange(len(df['Modèle']))
        bar_width = 0.35
        
        bars1 = ax2.bar(x - bar_width/2, df['mAP@50_%'], bar_width,
                       label='mAP@50 (%)', 
                       color=self.STYLE_CONFIG['theme_color'],
                       edgecolor=self.STYLE_CONFIG['bar_edge_color'])
        
        bars2 = ax2.bar(x + bar_width/2, df['GFLOPs'], bar_width,
                       label='GFLOPs', 
                       color=self.STYLE_CONFIG['accent_color'],
                       edgecolor=self.STYLE_CONFIG['bar_edge_color'])
        
        ax2.set_xlabel('Modèles')
        ax2.set_ylabel('Valeurs')
        ax2.set_xticks(x)
        ax2.set_xticklabels(df['Modèle'], rotation=45, ha='right')
        ax2.legend()
        
        # Application du style aux deux sous-graphiques
        for ax_sub in [ax1, ax2]:
            self.apply_style(ax_sub, "")
        
        fig.suptitle('Caractéristiques et performances des modèles', 
                    fontsize=self.STYLE_CONFIG['title_size'], 
                    fontweight='bold', y=0.98)
        
        self.save_plot(fig, "06_caracteristiques_modeles.png")
    
    def generate_all_graphs(self):
        """Génère tous les graphiques automatiquement"""
        print("🎨 GÉNÉRATION DES GRAPHIQUES AVEC STYLE UNIFIÉ")
        print("=" * 60)
        print(f"🎯 Thème: {self.STYLE_CONFIG['theme_color']}")
        print(f"🎯 Accent: {self.STYLE_CONFIG['accent_color']}")
        print("=" * 60)
        
        # Liste des graphiques à générer
        graphs = [
            ("Comparaison mAP@50", self.plot_comparison_map50),
            ("Métriques originales", self.plot_metrics_original),
            ("Métriques augmentées", self.plot_metrics_augmented),
            ("Performance par classe", self.plot_class_performance),
            ("F1-score par classe", self.plot_f1_by_class),
            ("Groupes fonctionnels", self.plot_functional_groups),
            ("Caractéristiques modèles", self.plot_model_characteristics),
        ]
        
        for graph_name, graph_func in graphs:
            print(f"📊 Création: {graph_name}")
            try:
                graph_func()
            except Exception as e:
                print(f"❌ Erreur avec {graph_name}: {e}")
        
        print("=" * 60)
        print("✅ TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS AVEC SUCCÈS!")
        print("📁 Fichiers sauvegardés dans le dossier courant")

# 🔧 UTILISATION ULTRA-SIMPLE
if __name__ == "__main__":
    # Initialisation du générateur
    generator = GraphiqueGenerator()
    
    # 🔽 PERSONNALISATION FACILE - MODIFIEZ UNIQUEMENT CES VALEURS 🔽
    generator.STYLE_CONFIG.update({
        'theme_color': '#2E75B6',      # Couleur principale
        'accent_color': '#ED7D31',     # Couleur secondaire  
        'success_color': '#70AD47',    # Couleur succès
        'warning_color': '#FFC000',    # Couleur avertissement
        'danger_color': '#C00000',     # Couleur danger
        'bg_color': '#FFFFFF',         # Fond des graphiques
        'font_family': 'Arial',        # Police
        'title_size': 18,              # Taille des titres
        'fig_width': 14,               # Largeur des graphiques
        'fig_height': 8,               # Hauteur des graphiques
    })
    
    # Lancement de la génération
    generator.generate_all_graphs()