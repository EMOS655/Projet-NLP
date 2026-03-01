import pandas as pd
import numpy as np
import os
import sys
import json
from scipy.stats import chi2_contingency

# --- GESTION DES CHEMINS ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from src.config import RESULTS_DIR
except ImportError:
    import config
    RESULTS_DIR = config.RESULTS_DIR

class StatAnalyzer:
    def __init__(self):
        self.input_file = os.path.join(RESULTS_DIR, "audit_results.csv")
        self.output_json = os.path.join(RESULTS_DIR, "statistical_conformity.json")

    def run_analysis(self):
        print("\n" + "="*70)
        print(" RÉSULTATS STATISTIQUES FINAUX")
        print("="*70)

        if not os.path.exists(self.input_file):
            print(f"[-] ERREUR : Le fichier '{self.input_file}' est introuvable.")
            return

        df = pd.read_csv(self.input_file)
        print(f"[+] Données analysées : {len(df)} lignes budgétaires.")

        # 1. Tableau de contingence et Chi-deux
        contingency = pd.crosstab(df['exercice'], df['pilier_predit'])
        chi2, p, dof, expected = chi2_contingency(contingency)

        # 2. Calcul des proportions pour le graphique Radar
        # On prend la moyenne des proportions sur tous les exercices pour la vue globale
        repartition_reelle = df['pilier_predit'].value_counts(normalize=True) * 100
        
        # Cibles théoriques SND30
        cibles_snd30 = {
            "Transformation structurelle": 30.0,
            "Capital humain": 25.0,
            "Gouvernance": 20.0,
            "Développement régional": 15.0,
            "Autre": 10.0
        }

        comparison_data = []
        for p_name in cibles_snd30.keys():
            # Données Réelles
            comparison_data.append({
                "pilier": p_name,
                "valeur": round(repartition_reelle.get(p_name, 0), 2),
                "type": "Réel (Loi de Finances)"
            })
            # Données Cibles
            comparison_data.append({
                "pilier": p_name,
                "valeur": cibles_snd30[p_name],
                "type": "Objectifs SND30"
            })

        # 3. Calcul du score de conformité (100 - erreur moyenne)
        diffs = [abs(repartition_reelle.get(k, 0) - v) for k, v in cibles_snd30.items()]
        score_conformite = max(0, 100 - np.mean(diffs))

        # 4. Interprétation
        interpretation = (
            "Le glissement sémantique est statistiquement significatif. "
            if p < 0.05 else "L'alignement est statistiquement stable. "
        )
        interpretation += "On observe une forte concentration sur le pilier " + df['pilier_predit'].mode()[0] + "."

        # 5. SAUVEGARDE POUR LE DASHBOARD
        stats_dashboard = {
            "conformity_score": round(score_conformite, 1),
            "p_value": float(p),
            "interpretation": interpretation,
            "comparison_data": comparison_data
        }

        with open(self.output_json, "w", encoding="utf-8") as f:
            json.dump(stats_dashboard, f, indent=4, ensure_ascii=False)

        print(f"[+] SUCCESS : Fichier JSON généré pour le Dashboard.")
        
        # --- Garder vos affichages console originaux ---
        print("\n[1] TEST DU CHI-DEUX :")
        print(f" - P-value : {p:.4e} ({'Significatif' if p < 0.05 else 'Non significatif'})")
        
        if 'cluster_id' in df.columns:
            print("\n[2] ANALYSE RAPIDE DES CLUSTERS :")
            print(df.groupby('cluster_id')['pilier_predit'].agg(pd.Series.mode).to_string())

        print("\n" + "="*70)

if __name__ == "__main__":
    analyzer = StatAnalyzer()
    analyzer.run_analysis()