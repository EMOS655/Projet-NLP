import json
import logging
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Configuration du chemin projet
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from src.utils import setup_logging, save_results, Timer

logger = setup_logging()

class BudgetStatisticalAnalyzer:
    """Analyse les résultats de classification pour en extraire des insights stratégiques."""
    
    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.comparison_file = self.results_dir / "comparison_piliers_2024_2025.json"

    def load_data(self) -> dict:
        if not self.comparison_file.exists():
            logger.error("Fichier de comparaison introuvable. Lancez classification.py d'abord.")
            return {}
        with open(self.comparison_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def analyze_evolution(self):
        """Calcule les variations absolues et relatives entre 2024 et 2025."""
        data = self.load_data()
        if not data: return

        # Transformation en DataFrame pour manipulation facile
        df = pd.DataFrame(data)
        df.columns = ['2024', '2025']
        
        # 1. Calcul des variations
        df['Variation_Absolue'] = df['2025'] - df['2024']
        # Calcul du pourcentage (avec gestion du zéro)
        df['Variation_Pourcentage'] = (df['Variation_Absolue'] / df['2024'].replace(0, np.nan) * 100).fillna(0)
        
        # 2. Calcul des parts de marché (Market Share des piliers)
        total_24 = df['2024'].sum()
        total_25 = df['2025'].sum()
        
        df['Part_2024_%'] = (df['2024'] / total_24 * 100).round(2)
        df['Part_2025_%'] = (df['2025'] / total_25 * 100).round(2)

        # 3. Indice de concentration (Herfindahl-Hirschman Index simplifié)
        # Plus l'indice est élevé, plus le budget est concentré sur peu de piliers
        hhi_24 = (df['Part_2024_%']**2).sum()
        hhi_25 = (df['Part_2025_%']**2).sum()

        stats_report = {
            "piliers_stats": df.to_dict(orient='index'),
            "metriques_globales": {
                "total_articles_2024": int(total_24),
                "total_articles_2025": int(total_25),
                "concentration_index_2024": float(hhi_24),
                "concentration_index_2025": float(hhi_25),
                "indice_diversification_evol": "Améliorée" if hhi_25 < hhi_24 else "Réduite"
            }
        }

        # Sauvegarde
        save_results(stats_report, "rapport_statistique_budget")
        
        self._print_summary(df, stats_report['metriques_globales'])
        return stats_report

    def _print_summary(self, df, metrics):
        """Affiche un résumé propre dans la console."""
        print("\n" + "="*60)
        print("📊 ANALYSE STATISTIQUE DE L'ALIGNEMENT SND30")
        print("="*60)
        print(df[['2024', '2025', 'Variation_Absolue', 'Part_2025_%']])
        print("-" * 60)
        print(f"🔹 Volume global : {metrics['total_articles_2025']} articles en 2025")
        print(f"🔹 Indice de concentration : {metrics['concentration_index_2025']:.2f}")
        print(f"🔹 Évolution de la diversité : {metrics['indice_diversification_evol']}")
        print("="*60)

if __name__ == "__main__":
    analyzer = BudgetStatisticalAnalyzer()
    analyzer.analyze_evolution()