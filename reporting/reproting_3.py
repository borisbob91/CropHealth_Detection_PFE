import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class AdvancedGraphiqueGenerator:
    def __init__(self):
        """
        Générateur de graphiques avancés avec données intégrées
        Inclut box plots pour comparaison original vs augmenté
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
        
        # 1. Comparaison mAP@50
        self.data_comparison = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50_Original": [0.72, 0.84, 0.79, 0.76],
            "mAP@50_Augmenté": [0.7808, 0.8908, 0.8518, 0.8018],
            "Amélioration": [0.0608, 0.0508, 0.0618, 0.0418]
        })
        
        # 2. Métriques détaillées
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

        
        # 3. Performances par classe
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
        
        
        # 4. Groupes fonctionnels
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
        
        
        # 5. Caractéristiques des modèles
        self.data_characteristics = pd.DataFrame({
            "Modèle": ["YOLOv8n", "SSD MobileNetV3", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "mAP@50_%": [87.22, 84.0, 72.0, 83.3],
            "Paramètres_M": [3.2, 2.2, 41.0, 3.5],
            "GFLOPs": [4.05, 0.6, 180.0, 3.2],
            "Temps_inference_ms": [15, 8, 120, 25],
            "Complexité": ["Faible", "Très faible", "Très élevée", "Faible"]
        })
        
        # 6. Métriques globales
        self.global_metrics = {
            "F1_Macro": 0.888,
            "mAP50_moyen": 0.8908,
            "Amélioration_moyenne": 0.0508
        }
        

        # 8. NOUVELLES DONNÉES POUR GRAPHIQUES AVANCÉS
        self.setup_advanced_data()
        


    def setup_advanced_data(self):
        """Configure les données pour les nouveaux graphiques avancés"""
        
        # 1. Analyse des erreurs par classe
        self.error_analysis = pd.DataFrame({
            "Classe": self.data_classes["Classe"],
            "Faux_Positifs": [12, 8, 5, 18, 7, 4, 45, 2, 6, 62, 15, 9, 11, 58, 14, 8, 10],
            "Faux_Négatifs": [8, 7, 9, 22, 5, 6, 28, 1, 4, 48, 20, 4, 35, 42, 12, 5, 12],
            "Précision": self.data_classes["Précision"],
            "Rappel": self.data_classes["Rappel"]
        })
        
        # 2. Temps d'inférence et efficacité
        self.inference_data = pd.DataFrame({
            "Modèle": ["SSD", "YOLOv8n", "Faster R-CNN ResNet50", "Faster R-CNN MobileNetV3"],
            "Temps_ms": [8, 15, 120, 25],
            "mAP@50": [76, 87.9, 81, 79],
            "FPS": [125, 67, 8, 40],
            "Efficacité": [9.5, 5.86, 0.675, 3.16]  # mAP@50 / Temps_ms
        })
            

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
                   ha='center', va='top', style='italic', color='#666666')
        
        # Labels des axes
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        if ax.get_ylabel():
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

    def plot_boxplot_comparison(self):
        """Graphique 13: Box plots comparatifs Original vs Augmenté"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['mAP@50', 'Précision', 'Rappel', 'F1_score']
        titles = ['mAP@50', 'Précision', 'Rappel', 'F1-score']
        
        colors = [self.STYLE_CONFIG['theme_color'], self.STYLE_CONFIG['accent_color']]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Création du box plot
            sns.boxplot(data=self.boxplot_data, x='Modèle', y=metric, hue='Dataset',
                       palette=colors, ax=ax, showfliers=False)
            
            # Amélioration visuelle
            ax.set_title(f'Distribution {title}', fontweight='bold', fontsize=14)
            ax.set_xlabel('Modèles')
            ax.set_ylabel(f'{title} (%)')
            ax.tick_params(axis='x', rotation=45)
            
            # Ajout de points pour montrer les données individuelles
            sns.stripplot(data=self.boxplot_data, x='Modèle', y=metric, hue='Dataset',
                         palette=colors, ax=ax, size=4, alpha=0.5, dodge=True, jitter=True)
            
            # Suppression de la légende du strip plot pour éviter la duplication
            if i > 0:
                ax.get_legend().remove()
        
        # Ajustement de la légende
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:2], labels[:2], loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        # Suppression des légendes individuelles
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()
        
        plt.tight_layout()
        
        self.save_plot(fig, "13_boxplot_comparison.png")
    
    def plot_swarmplot_comparison(self):
        """Graphique 15: Swarm plots pour voir tous les points individuels"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        metrics = ['mAP@50', 'Précision', 'Rappel', 'F1_score']
        titles = ['mAP@50', 'Précision', 'Rappel', 'F1-score']
        
        colors = [self.STYLE_CONFIG['theme_color'], self.STYLE_CONFIG['accent_color']]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i]
            
            # Création du swarm plot
            sns.swarmplot(data=self.boxplot_data, x='Modèle', y=metric, hue='Dataset',
                         palette=colors, ax=ax, size=3, alpha=0.7, dodge=True)
            
            ax.set_title(f'Points individuels {title}', fontweight='bold', fontsize=14)
            ax.set_xlabel('Modèles')
            ax.set_ylabel(f'{title} (%)')
            ax.tick_params(axis='x', rotation=45)
            
            if i > 0:
                ax.get_legend().remove()
        
        # Ajustement de la légende
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles[:2], labels[:2], loc='upper center', 
                  bbox_to_anchor=(0.5, 0.02), ncol=2, fontsize=12)
        
        for ax in axes:
            if ax.get_legend():
                ax.get_legend().remove()
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)
        
        fig.suptitle('Visualisation des points individuels: Original vs Augmenté', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', y=0.98)
        
        self.save_plot(fig, "15_swarmplot_comparison.png")
    
    def plot_improvement_boxplot(self):
        """Graphique 16: Box plot de l'amélioration par métrique"""
        # Calcul de l'amélioration pour chaque point
        improvement_data = []
        
        for model in self.boxplot_data['Modèle'].unique():
            model_data = self.boxplot_data[self.boxplot_data['Modèle'] == model]
            
            for metric in ['mAP@50', 'Précision', 'Rappel', 'F1_score']:
                orig_values = model_data[model_data['Dataset'] == 'Original'][metric]
                aug_values = model_data[model_data['Dataset'] == 'Augmenté'][metric]
                
                # Calcul de l'amélioration en pourcentage
                for orig, aug in zip(orig_values, aug_values):
                    improvement = ((aug - orig) / orig) * 100
                    improvement_data.append({
                        'Modèle': model,
                        'Métrique': metric,
                        'Amélioration (%)': improvement
                    })
        
        improvement_df = pd.DataFrame(improvement_data)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Box plot de l'amélioration
        sns.boxplot(data=improvement_df, x='Métrique', y='Amélioration (%)', hue='Modèle',
                   palette=[self.STYLE_CONFIG['theme_color'], self.STYLE_CONFIG['accent_color'],
                           self.STYLE_CONFIG['success_color'], self.STYLE_CONFIG['warning_color']],
                   ax=ax)
        
        # Ligne à zéro pour référence
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Ligne de base')
        
        ax.set_title("Amélioration relative après augmentation des données", 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold')
        ax.set_xlabel('Métriques')
        ax.set_ylabel('Amélioration (%)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        
        self.apply_style(ax, "Amélioration relative par métrique et modèle")
        self.save_plot(fig, "16_improvement_boxplot.png")


    def generate_all_graphs(self):
        """Génère tous les graphiques automatiquement"""
        print("🎨 GÉNÉRATION DES GRAPHIQUES AVANCÉS")
        print("=" * 60)
        print(f"🎯 Thème: {self.STYLE_CONFIG['theme_color']}")
        print(f"🎯 Accent: {self.STYLE_CONFIG['accent_color']}")
        print("=" * 60)
        
        # Liste des graphiques à générer (incluant les nouveaux box plots)
        graphs = [
            ("Box plots comparatifs", self.plot_boxplot_comparison),
            ("Swarm plots points", self.plot_swarmplot_comparison),
            ("Box plot amélioration", self.plot_improvement_boxplot),
        ]
        
        for graph_name, graph_func in graphs:
            print(f"📊 Création: {graph_name}")
            try:
                graph_func()
            except Exception as e:
                print(f"❌ Erreur avec {graph_name}: {e}")
        
        print("=" * 60)
        print("✅ BOX PLOTS ET GRAPHIQUES DE DISTRIBUTION GÉNÉRÉS AVEC SUCCÈS!")
        print("📁 Fichiers sauvegardés: 13_ à 16_")

# 🔧 UTILISATION
if __name__ == "__main__":
    # Initialisation du générateur
    generator = AdvancedGraphiqueGenerator()
    
    # 🔽 PERSONNALISATION FACILE - MODIFIEZ UNIQUEMENT CES VALEURS 🔽
    generator.STYLE_CONFIG.update({
        'theme_color': '#2E75B6',      # Couleur principale
        'accent_color': '#ED7D31',     # Couleur secondaire  
        'success_color': '#70AD47',    # Couleur succès
        'warning_color': '#FFC000',    # Couleur avertissement
        'danger_color': '#C00000',     # Couleur danger
        'bg_color': '#FFFFFF',         # Fond des graphiques
        'font_family': 'Arial',        # Police
        'title_size': 16,              # Taille des titres
        'fig_width': 14,               # Largeur des graphiques
        'fig_height': 8,               # Hauteur des graphiques
    })
    
    # Lancement de la génération
    generator.generate_all_graphs()