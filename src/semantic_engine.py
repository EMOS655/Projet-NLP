import os
import sys
import json
import torch
import pandas as pd
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util

# --- 1. GESTION DES CHEMINS ET IMPORTS DE CONFIG ---
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

try:
    import config
    SND30_DESCRIPTIONS = config.SND30_DESCRIPTIONS
    SENTENCE_BERT_MODEL = config.SENTENCE_BERT_MODEL
    DEVICE = config.DEVICE
    RESULTS_DIR = config.RESULTS_DIR
except ImportError:
    from src.config import SND30_DESCRIPTIONS, SENTENCE_BERT_MODEL, DEVICE, RESULTS_DIR

# --- 2. CLASSE DE TRAITEMENT SÉMANTIQUE ---
class SemanticAudit:
    def __init__(self):
        print(f"[#] Initialisation : {SENTENCE_BERT_MODEL} sur {DEVICE}")
        self.model = SentenceTransformer(SENTENCE_BERT_MODEL, device=DEVICE)
        
        # Préparation des piliers SND30
        self.piliers_names = list(SND30_DESCRIPTIONS.keys())
        descriptions = list(SND30_DESCRIPTIONS.values())
        self.piliers_embeddings = self.model.encode(descriptions, convert_to_tensor=True)

    def classify_and_cluster(self, df, n_clusters=5):
        """
        Réalise la classification (SND30) ET le clustering (K-Means)
        """
        if df is None or df.empty:
            return df
            
        print(f"[#] Encodage sémantique de {len(df)} lignes...")
        textes = df['libelle_projet'].astype(str).tolist()
        embeddings = self.model.encode(textes, convert_to_tensor=True)
        
        # --- PARTIE 1 : CLASSIFICATION SND30 (Approche supervisée) ---
        print("[#] Classification par rapport aux piliers SND30...")
        scores = util.cos_sim(embeddings, self.piliers_embeddings)
        best_indices = torch.argmax(scores, dim=1).tolist()
        
        df['pilier_predit'] = [self.piliers_names[i] for i in best_indices]
        df['score_similarite'] = torch.max(scores, dim=1).values.tolist()
        
        # --- PARTIE 2 : CLUSTERING K-MEANS (Approche non-supervisée) ---
        print(f"[#] Clustering K-Means pour détecter des thèmes émergents (k={n_clusters})...")
        # On repasse en CPU/Numpy pour Scikit-Learn
        embeddings_np = embeddings.cpu().detach().numpy()
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['cluster_id'] = kmeans.fit_predict(embeddings_np)
        
        return df

# --- 3. POINT D'ENTRÉE PRINCIPAL ---
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  AUDIT & CLUSTERING")
    print("="*60)
    
    # Import de l'extracteur (après s'être assuré que le path est correct)
    try:
        from extrator import PDFExtractor
    except ImportError:
        from src.extrator import PDFExtractor

    # Initialisation des moteurs
    extractor = PDFExtractor()
    audit = SemanticAudit()

    # Étape 1 : Extraction
    print("\n[Étape 1] Extraction des données des PDF...")
    df_projets = extractor.process_all_pdfs()

    # Étape 2 : Traitement (Classification + Clustering)
    if df_projets is not None and not df_projets.empty:
        print("\n[Étape 2] Analyse sémantique et Clustering...")
        df_final = audit.classify_and_cluster(df_projets, n_clusters=5)
        
        # Étape 3 : Sauvegarde des résultats
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        save_path = os.path.join(RESULTS_DIR, "audit_results.csv")
        df_final.to_csv(save_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "-"*60)
        print(f"[SUCCESS] Audit et Clustering terminés !")
        print(f"[PATH] Fichier généré : {save_path}")
        print(f"[INFO] Colonne 'pilier_predit' : Thèmes SND30 officiels")
        print(f"[INFO] Colonne 'cluster_id'    : Groupements thématiques automatiques")
        print("-" * 60)
        
        # Aperçu
        print(df_final[['exercice', 'libelle_projet', 'pilier_predit', 'cluster_id']].head(10))
    else:
        print("\n[!] ERREUR : Aucun projet trouvé. Vérifiez le dossier data/raw/")

    print("\n[Prochaine étape] : python src/classifier.py")
    print("="*60)