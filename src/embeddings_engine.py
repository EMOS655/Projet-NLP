import torch
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import os
import sys

# --- CONFIGURATION DES CHEMINS ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    from src.config import SENTENCE_BERT_MODEL, DEVICE, RESULTS_DIR
except ImportError:
    import config
    SENTENCE_BERT_MODEL = config.SENTENCE_BERT_MODEL
    DEVICE = config.DEVICE
    RESULTS_DIR = config.RESULTS_DIR

class SemanticComparison:
    def __init__(self):
        print(f"[#] Initialisation de l'Audit Sémantique : {SENTENCE_BERT_MODEL}")
        self.model = SentenceTransformer(SENTENCE_BERT_MODEL, device=DEVICE)

    def compute_ruptures(self, df_2024, df_2025, threshold=0.70):
        """
        Compare les articles de 2025 à ceux de 2024 pour identifier les ruptures de discours.
        """
        if df_2024.empty or df_2025.empty:
            print("[!] Erreur : Un des DataFrames est vide.")
            return None

        # 1. Identification intelligente de la colonne de texte
        # On cherche 'libelle_projet' (votre colonne par défaut) ou 'contenu_nettoye'
        cols_lf = df_2024.columns.tolist()
        text_col = next((c for c in ['libelle_projet', 'contenu_nettoye', 'contenu'] if c in cols_lf), None)
        
        if not text_col:
            print(f"[!] Colonne de texte introuvable parmi : {cols_lf}")
            return None

        print(f"[#] Utilisation de la colonne : '{text_col}'")

        # 2. Préparation et Encodage (Gestion des valeurs nulles)
        texts_24 = df_2024[text_col].fillna("Information non spécifiée").astype(str).tolist()
        texts_25 = df_2025[text_col].fillna("Information non spécifiée").astype(str).tolist()

        print(f"[#] Encodage des articles ({len(texts_24)} en 2024 vs {len(texts_25)} en 2025)...")
        emb_2024 = self.model.encode(texts_24, convert_to_tensor=True)
        emb_2025 = self.model.encode(texts_25, convert_to_tensor=True)

        # 3. Calcul de la similarité cosinus (Matrice 2025 x 2024)
        # Chaque ligne de 2025 est comparée à TOUTES les lignes de 2024
        cosine_scores = util.cos_sim(emb_2025, emb_2024)

        ruptures = []
        for i in range(len(df_2025)):
            # Trouver le score de similarité le plus élevé dans 2024 pour l'article i de 2025
            max_score, match_idx = torch.max(cosine_scores[i], dim=0)
            score = max_score.item()
            
            is_rupture = score < threshold
            
            ruptures.append({
                'exercice': 2025,
                'article_2025': texts_25[i][:200], # On garde un extrait pour le rapport
                'meilleur_match_2024': texts_24[match_idx.item()][:200],
                'score_similarite': round(score, 4),
                'diagnostic': "RUPTURE" if is_rupture else "CONTINUITÉ"
            })

        return pd.DataFrame(ruptures)

if __name__ == "__main__":
    # Chargement des données issues de l'audit précédent
    path_audit = os.path.join(RESULTS_DIR, "audit_results.csv")
    
    if os.path.exists(path_audit):
        df_full = pd.read_csv(path_audit)
        
        # Nettoyage des noms de colonnes (espaces invisibles)
        df_full.columns = df_full.columns.str.strip()
        
        # Séparation des exercices
        df_24 = df_full[df_full['exercice'] == 2024].copy()
        df_25 = df_full[df_full['exercice'] == 2025].copy()
        
        if df_24.empty or df_25.empty:
            print(f"[!] Données insuffisantes pour comparer 2024 ({len(df_24)}) et 2025 ({len(df_25)})")
        else:
            comparer = SemanticComparison()
            df_results = comparer.compute_ruptures(df_24, df_25)
            
            if df_results is not None:
                save_path = os.path.join(RESULTS_DIR, "ruptures_semantiques.csv")
                df_results.to_csv(save_path, index=False, encoding='utf-8-sig')
                
                # Statistiques rapides
                n_ruptures = len(df_results[df_results['diagnostic'] == 'RUPTURE'])
                print("\n" + "="*60)
                print(f" AUDIT PAR PLONGEMENTS TERMINÉ")
                print(f" - Ruptures de discours détectées : {n_ruptures}")
                print(f" - Taux de continuité sémantique : {((len(df_25)-n_ruptures)/len(df_25)):.1%}")
                print(f" - Fichier généré : {save_path}")
                print("="*60)
    else:
        print(f"[!] Erreur : {path_audit} introuvable. Lancez d'abord votre pipeline d'audit.")