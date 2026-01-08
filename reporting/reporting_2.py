import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class EvaluationDataProcessor:
    def __init__(self):
        """Classe pour traiter et visualiser les données d'évaluation détaillées"""
        self.setup_style()
        self.setup_data()
    
    def setup_style(self):
        """Configure le style des graphiques"""
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.STYLE_CONFIG = {
            'theme_color': '#2E75B6',
            'accent_color': '#ED7D31', 
            'success_color': '#70AD47',
            'warning_color': '#FFC000',
            'danger_color': '#FF0000',
            'bg_color': '#FFFFFF',
            'font_family': 'Arial',
            'title_size': 16,
            'label_size': 12,
            'fig_width': 12,
            'fig_height': 8,
            'dpi': 300
        }
    
    def setup_data(self):
        """Intègre les données d'évaluation détaillées"""
        
        # Données brutes fournies
        self.evaluation_data = pd.DataFrame({
            'Classe': [
                'all', 'A. flava', 'B. tabaci', 'Coccinelle', 'Degat Jassides', 
                'Dysdercus spp', 'Earias spp', 'Effet phyto', 'G. spodoctera', 
                'H. amirgera', 'Jasside', 'Larve coccinelle', 'Larve syrphe', 
                'P. gossypiella', 'Puceron', 'S. derogata', 'S. frugiperda', 'Scarabees'
            ],
            'Images': [2244, 75, 45, 156, 245, 98, 148, 186, 97, 189, 140, 184, 51, 90, 86, 258, 145, 100],
            'Instances': [3328, 75, 135, 156, 474, 98, 148, 206, 97, 189, 703, 229, 51, 90, 131, 301, 145, 100],
            'Précision': [0.871, 0.98, 0.964, 0.814, 0.733, 0.978, 0.981, 0.503, 0.996, 0.986, 0.705, 0.941, 0.959, 0.925, 0.505, 0.905, 0.933, 0.995],
            'Rappel': [0.878, 0.987, 0.941, 0.872, 0.662, 0.98, 0.966, 0.738, 1.0, 0.989, 0.745, 0.991, 1.0, 0.967, 0.26, 0.94, 1.0, 0.88],
            'mAP@50': [0.879, 0.978, 0.966, 0.87, 0.739, 0.974, 0.992, 0.698, 0.995, 0.994, 0.707, 0.992, 0.995, 0.969, 0.286, 0.92, 0.984, 0.886],
            'mAP@50-95': [0.492, 0.477, 0.481, 0.415, 0.392, 0.56, 0.583, 0.388, 0.812, 0.563, 0.266, 0.55, 0.724, 0.327, 0.135, 0.544, 0.62, 0.529]
        })
        
        # Calcul des métriques supplémentaires
        self.calculate_additional_metrics()
    
    def calculate_additional_metrics(self):
        """Calcule des métriques supplémentaires pour l'analyse"""
        
        # F1-score
        self.evaluation_data['F1-score'] = 2 * (
            self.evaluation_data['Précision'] * self.evaluation_data['Rappel']
        ) / (self.evaluation_data['Précision'] + self.evaluation_data['Rappel'])
        
        # Ratio Instances/Images (densité)
        self.evaluation_data['Densité'] = (
            self.evaluation_data['Instances'] / self.evaluation_data['Images']
        )
        
        # Écart entre mAP@50 et mAP@50-95
        self.evaluation_data['Écart_mAP'] = (
            self.evaluation_data['mAP@50'] - self.evaluation_data['mAP@50-95']
        )
        
        # Catégorie de performance
        conditions = [
            self.evaluation_data['mAP@50'] >= 0.9,
            self.evaluation_data['mAP@50'] >= 0.7,
            self.evaluation_data['mAP@50'] >= 0.5,
            self.evaluation_data['mAP@50'] < 0.5
        ]
        choices = ['Excellent', 'Bon', 'Moyen', 'Faible']
        self.evaluation_data['Performance'] = np.select(conditions, choices, default='Faible')
    
    def create_excel_report(self, filename="evaluation_detaillees.xlsx"):
        """Crée un fichier Excel complet avec les données d'évaluation"""
        
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            
            # Feuille 1: Données brutes
            self.evaluation_data.to_excel(writer, sheet_name='Données Brutes', index=False)
            
            # Feuille 2: Statistiques descriptives
            stats_df = self.evaluation_data.describe()
            stats_df.to_excel(writer, sheet_name='Statistiques')
            
            # Feuille 3: Performance par classe (triée)
            perf_df = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
            perf_df = perf_df.sort_values('mAP@50', ascending=False)
            perf_df.to_excel(writer, sheet_name='Performance Classes', index=False)
            
            # Feuille 4: Analyse par catégorie de performance
            performance_summary = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].groupby('Performance').agg({
                'Classe': 'count',
                'mAP@50': ['mean', 'std'],
                'mAP@50-95': ['mean', 'std'],
                'F1-score': 'mean',
                'Images': 'sum',
                'Instances': 'sum'
            }).round(3)
            performance_summary.to_excel(writer, sheet_name='Analyse Performance')
            
            # Feuille 5: Top et Flop
            top_5 = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].nlargest(5, 'mAP@50')[['Classe', 'mAP@50', 'mAP@50-95', 'F1-score']]
            flop_5 = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].nsmallest(5, 'mAP@50')[['Classe', 'mAP@50', 'mAP@50-95', 'F1-score']]
            
            top_5.to_excel(writer, sheet_name='Top 5 Classes', index=False)
            flop_5.to_excel(writer, sheet_name='Flop 5 Classes', index=False)
        
        print(f"✅ Fichier Excel créé: {filename}")
        return filename

    def save_individual_plot(self, fig, filename):
        """Sauvegarde un graphique individuel avec haute qualité"""
        fig.savefig(
            filename, 
            dpi=self.STYLE_CONFIG['dpi'], 
            bbox_inches='tight', 
            facecolor=self.STYLE_CONFIG['bg_color'],
            transparent=False
        )
        plt.close(fig)
        print(f"✅ {filename}")

    def plot_map50_by_class(self):
        """Graphique 1: Performance mAP@50 par classe (barres horizontales)"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        plot_data_sorted = plot_data.sort_values('mAP@50', ascending=True)
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        bars = ax.barh(plot_data_sorted['Classe'], plot_data_sorted['mAP@50'], 
                       color=[self.STYLE_CONFIG['success_color'] if x >= 0.7 else 
                             self.STYLE_CONFIG['warning_color'] if x >= 0.5 else 
                             self.STYLE_CONFIG['danger_color'] for x in plot_data_sorted['mAP@50']],
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('mAP@50', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_title('Performance mAP@50 par Classe', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.set_xlim(0, 1.05)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Grille et style
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor(self.STYLE_CONFIG['bg_color'])
        
        self.save_individual_plot(fig, "01_map50_par_classe.png")

    def plot_precision_vs_recall(self):
        """Graphique 2: Scatter plot Précision vs Rappel"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        scatter = ax.scatter(plot_data['Précision'], plot_data['Rappel'], 
                             s=plot_data['mAP@50']*200, alpha=0.7,
                             c=plot_data['mAP@50'], cmap='viridis', edgecolors='black', linewidth=0.5)
        
        # Annotation des points problématiques
        for i, row in plot_data.iterrows():
            if row['Précision'] < 0.6 or row['Rappel'] < 0.6 or row['mAP@50'] < 0.5:
                ax.annotate(row['Classe'], (row['Précision'], row['Rappel']),
                           xytext=(8, 8), textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel('Précision', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_ylabel('Rappel', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        
        # Ligne de référence idéale
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Précision = Rappel')
        
        # Barre de couleur
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('mAP@50', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        ax.set_title('Précision vs Rappel par Classe', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        self.save_individual_plot(fig, "02_precision_vs_rappel.png")

    def plot_map_comparison(self):
        """Graphique 3: Comparaison mAP@50 vs mAP@50-95"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        x = np.arange(len(plot_data))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, plot_data['mAP@50'], width, 
                       label='mAP@50', alpha=0.8, color=self.STYLE_CONFIG['theme_color'],
                       edgecolor='black', linewidth=0.5)
        
        bars2 = ax.bar(x + width/2, plot_data['mAP@50-95'], width, 
                       label='mAP@50-95', alpha=0.8, color=self.STYLE_CONFIG['accent_color'],
                       edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Classes', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_ylabel('Score mAP', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(plot_data['Classe'], rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=self.STYLE_CONFIG['label_size'])
        ax.set_ylim(0, 1.1)
        
        ax.set_title('Comparaison mAP@50 vs mAP@50-95', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        self.save_individual_plot(fig, "03_comparaison_map.png")


    def plot_instances_distribution(self):
        """Graphique 5: Distribution des instances par classe"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        plot_data_sorted = plot_data.sort_values('Instances', ascending=True)
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        bars = ax.barh(plot_data_sorted['Classe'], plot_data_sorted['Instances'], 
                       color=self.STYLE_CONFIG['theme_color'], alpha=0.8,
                       edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Nombre d'Instances", fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_title("Distribution des Instances par Classe", 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 5, bar.get_y() + bar.get_height()/2, f'{int(width)}', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_facecolor(self.STYLE_CONFIG['bg_color'])
        
        self.save_individual_plot(fig, "05_distribution_instances.png")

    def plot_performance_vs_instances(self):
        """Graphique 6: Performance vs Nombre d'instances"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        scatter = ax.scatter(plot_data['Instances'], plot_data['mAP@50'], 
                             s=100, alpha=0.7, color=self.STYLE_CONFIG['accent_color'],
                             edgecolors='black', linewidth=0.5)
        
        # Régression linéaire
        z = np.polyfit(plot_data['Instances'], plot_data['mAP@50'], 1)
        p = np.poly1d(z)
        ax.plot(plot_data['Instances'], p(plot_data['Instances']), "r--", alpha=0.8, linewidth=2)
        
        # Annotation des points intéressants
        for i, row in plot_data.iterrows():
            if row['mAP@50'] < 0.5 or row['Instances'] > 400 or row['mAP@50'] > 0.95:
                ax.annotate(row['Classe'], (row['Instances'], row['mAP@50']),
                           xytext=(8, 8), textcoords='offset points', fontsize=9,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel("Nombre d'Instances", fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_ylabel('mAP@50', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        
        # Coefficient de corrélation
        correlation = plot_data['Instances'].corr(plot_data['mAP@50'])
        ax.text(0.05, 0.95, f'Corrélation: {correlation:.3f}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
        
        ax.set_title('Performance vs Taille du Dataset', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        self.save_individual_plot(fig, "06_performance_vs_instances.png")

    def plot_f1_score_analysis(self):
        """Graphique 7: Analyse du F1-score par classe"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        plot_data_sorted = plot_data.sort_values('F1-score', ascending=True)
        
        fig, ax = plt.subplots(figsize=(self.STYLE_CONFIG['fig_width'], self.STYLE_CONFIG['fig_height']))
        
        bars = ax.barh(plot_data_sorted['Classe'], plot_data_sorted['F1-score'], 
                       color=[self.STYLE_CONFIG['success_color'] if x >= 0.8 else 
                             self.STYLE_CONFIG['warning_color'] if x >= 0.6 else 
                             self.STYLE_CONFIG['danger_color'] for x in plot_data_sorted['F1-score']],
                       alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('F1-score', fontsize=self.STYLE_CONFIG['label_size'], fontweight='bold')
        ax.set_title('F1-score par Classe', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        ax.set_xlim(0, 1.05)
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.3f}', 
                    ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Ligne de référence
        ax.axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Seuil excellent (0.8)')
        ax.axvline(x=0.6, color='orange', linestyle='--', alpha=0.5, label='Seuil acceptable (0.6)')
        
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend()
        
        self.save_individual_plot(fig, "07_f1_score_analysis.png")

    def plot_performance_categories(self):
        """Graphique 8: Répartition des classes par catégorie de performance"""
        plot_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all'].copy()
        
        performance_counts = plot_data['Performance'].value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = [self.STYLE_CONFIG['success_color'], self.STYLE_CONFIG['warning_color'], 
                 self.STYLE_CONFIG['accent_color'], self.STYLE_CONFIG['danger_color']]
        
        wedges, texts, autotexts = ax.pie(performance_counts.values, labels=performance_counts.index, 
                                         autopct='%1.1f%%', colors=colors, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
        
        # Améliorer l'apparence des pourcentages
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Répartition des Classes par Catégorie de Performance', 
                    fontsize=self.STYLE_CONFIG['title_size'], fontweight='bold', pad=20)
        
        self.save_individual_plot(fig, "08_categories_performance.png")

    def generate_all_individual_plots(self):
        """Génère tous les graphiques individuels"""
        
        print("🎨 GÉNÉRATION DES GRAPHIQUES INDIVIDUELS")
        print("=" * 50)
        
        plots = [
            ("mAP@50 par classe", self.plot_map50_by_class),
            ("Précision vs Rappel", self.plot_precision_vs_recall),
            ("Comparaison mAP", self.plot_map_comparison),
            ("Matrice de corrélation", self.plot_correlation_heatmap),
            ("Distribution instances", self.plot_instances_distribution),
            ("Performance vs Instances", self.plot_performance_vs_instances),
            ("Analyse F1-score", self.plot_f1_score_analysis),
            ("Catégories performance", self.plot_performance_categories),
        ]
        
        for plot_name, plot_func in plots:
            print(f"📊 Création: {plot_name}")
            try:
                plot_func()
            except Exception as e:
                print(f"❌ Erreur avec {plot_name}: {e}")
        
        print("=" * 50)
        print("✅ TOUS LES GRAPHIQUES ONT ÉTÉ GÉNÉRÉS INDIVIDUELLEMENT!")

    def generate_comprehensive_report(self):
        """Génère un rapport complet avec Excel et tous les graphiques individuels"""
        
        print("📊 GÉNÉRATION DU RAPPORT D'ÉVALUATION COMPLET")
        print("=" * 50)
        
        # Création du fichier Excel
        excel_file = self.create_excel_report()
        
        # Génération de tous les graphiques individuels
        self.generate_all_individual_plots()
        
        # Affichage des statistiques clés
        self.display_key_metrics()
        
        print("=" * 50)
        print("✅ RAPPORT COMPLET GÉNÉRÉ AVEC SUCCÈS!")
        
        return excel_file
    
    def display_key_metrics(self):
        """Affiche les métriques clés dans la console"""
        
        overall = self.evaluation_data[self.evaluation_data['Classe'] == 'all'].iloc[0]
        classes_data = self.evaluation_data[self.evaluation_data['Classe'] != 'all']
        
        print("\n📈 MÉTRIQUES CLÉS DU MODÈLE:")
        print(f"   mAP@50 Global: {overall['mAP@50']:.3f}")
        print(f"   mAP@50-95 Global: {overall['mAP@50-95']:.3f}")
        print(f"   Précision Moyenne: {classes_data['Précision'].mean():.3f}")
        print(f"   Rappel Moyen: {classes_data['Rappel'].mean():.3f}")
        print(f"   F1-score Moyen: {classes_data['F1-score'].mean():.3f}")
        
        print(f"\n📊 DISTRIBUTION DES PERFORMANCES:")
        perf_counts = classes_data['Performance'].value_counts()
        for perf, count in perf_counts.items():
            print(f"   {perf}: {count} classes")
        
        print(f"\n🎯 TOP 3 CLASSES:")
        top_3 = classes_data.nlargest(3, 'mAP@50')[['Classe', 'mAP@50']]
        for _, row in top_3.iterrows():
            print(f"   {row['Classe']}: {row['mAP@50']:.3f}")
        
        print(f"\n⚠️  CLASSES À AMÉLIORER:")
        flop_3 = classes_data.nsmallest(3, 'mAP@50')[['Classe', 'mAP@50']]
        for _, row in flop_3.iterrows():
            print(f"   {row['Classe']}: {row['mAP@50']:.3f}")

# 🔧 UTILISATION SIMPLE
if __name__ == "__main__":
    # Initialisation du processeur
    processor = EvaluationDataProcessor()
    
    # Génération du rapport complet avec graphiques individuels
    processor.generate_all_individual_plots()