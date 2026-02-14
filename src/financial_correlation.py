import pandas as pd
import os
import sys
import re

# --- GESTION DES CHEMINS ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from src.config import RESULTS_DIR
except ImportError:
    import config
    RESULTS_DIR = config.RESULTS_DIR

class FinancialCorrelation:
    def __init__(self):
        self.input_file = os.path.join(RESULTS_DIR, "audit_results.csv")

    def extract_realistic_amount(self, text):
        """
        Extrait des montants financiers réalistes (1M à 500 Mds FCFA).
        Élimine le bruit (numéros de tel, codes projets, dates).
        """
        # Regex pour capturer des nombres longs avec séparateurs (espaces ou points)
        # On cherche des séquences de 7 à 12 chiffres (millions à centaines de milliards)
        matches = re.findall(r'\d[\d\s\.]{6,14}', str(text))
        
        valid_amounts = []
        for match in matches:
            # Nettoyage des caractères non numériques
            clean_val = re.sub(r'[^\d]', '', match)
            if clean_val:
                try:
                    val = float(clean_val)
                    # FILTRE ÉCONOMIQUE ISSEA :
                    # On ne garde que ce qui ressemble à un budget de projet (1 Million < X < 500 Milliards)
                    if 1_000_000 <= val <= 500_000_000_000:
                        valid_amounts.append(val)
                except ValueError:
                    continue
        
        # On retourne le montant le plus élevé trouvé dans la ligne (souvent le budget total)
        return max(valid_amounts) if valid_amounts else 0.0

    def run_analysis(self):
        print("\n" + "="*70)
        print("   ISSEA - ANALYSE DE LA COHÉRENCE FINANCIÈRE (SND30)")
        print("="*70)

        if not os.path.exists(self.input_file):
            print(f"[-] Erreur : Le fichier {self.input_file} est introuvable.")
            return

        df = pd.read_csv(self.input_file)
        
        print(f"[#] Analyse de {len(df)} lignes pour extraction financière...")
        
        # 1. Extraction filtrée
        df['montant_filtre'] = df['libelle_projet'].apply(self.extract_realistic_amount)

        # 2. Agrégation par Pilier et par Exercice
        # On calcule le nombre de projets et la somme des budgets
        financial_summary = df.groupby(['exercice', 'pilier_predit']).agg({
            'libelle_projet': 'count',
            'montant_filtre': 'sum'
        }).rename(columns={
            'libelle_projet': 'nb_projets', 
            'montant_filtre': 'budget_total_fcfa'
        })

        # 3. Calcul du coût moyen par projet
        financial_summary['cout_moyen_fcfa'] = (
            financial_summary['budget_total_fcfa'] / financial_summary['nb_projets']
        ).round(0)

        # 4. Affichage formatté (plus lisible que la notation scientifique)
        pd.options.display.float_format = '{:,.0f}'.format
        
        print("\n[RÉSULTATS] ALLOCATION BUDGÉTAIRE ESTIMÉE PAR PILIER :")
        print("-" * 80)
        print(financial_summary.sort_values(by=['exercice', 'budget_total_fcfa'], ascending=[True, False]))
        print("-" * 80)

        # 5. Calcul de la concentration financière
        total_global = df['montant_filtre'].sum()
        if total_global > 0:
            top_pilier = df.groupby('pilier_predit')['montant_filtre'].sum().idxmax()
            concentration = (df.groupby('pilier_predit')['montant_filtre'].sum().max() / total_global) * 100
            print(f"\n[CONSTAT] Priorité financière absolue : '{top_pilier}'")
            print(f"          Il capte environ {concentration:.1f}% de la masse financière extraite.")
        else:
            print("\n[!] Alerte : Aucun montant financier valide n'a été extrait. Vérifiez le format du texte.")

        # Sauvegarde
        output_path = os.path.join(RESULTS_DIR, "financial_analysis_clean.csv")
        financial_summary.to_csv(output_path)
        print(f"\n[+] Analyse sauvegardée avec succès : {output_path}")
        print("="*70)

if __name__ == "__main__":
    correlator = FinancialCorrelation()
    correlator.run_analysis()