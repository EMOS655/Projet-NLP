import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
from typing import List, Dict, Tuple, Union
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Import de sentence_transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers non installé. Tapez : pip install sentence-transformers")

import config
from src.utils import setup_logging, save_results, Timer, validate_similarity_score

# Setup logging
logger = setup_logging()

class SemanticAnalyzer:
    """Classe pour l'analyse sémantique avec embeddings (Optimisée CPU)"""
    
    def __init__(self, model_name: str = None):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers est requis.")
        
        self.model_name = model_name or config.SENTENCE_BERT_MODEL
        self.device = "cpu"
        
        logger.info(f"Chargement du modèle sur {self.device.upper()} : {self.model_name}")
        with Timer(f"Chargement du modèle {self.model_name}"):
            self.model = SentenceTransformer(self.model_name, device=self.device)
        
        logger.info(f"✓ Modèle chargé avec succès")
    
    def compute_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Calcule les embeddings pour une liste de textes"""
        if not texts:
            return np.array([])
        
        logger.info(f"Calcul des embeddings pour {len(texts)} textes...")
        with Timer(f"Encodage de {len(texts)} textes"):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True 
            )
        return embeddings

    def detect_semantic_drift(
        self, 
        texts_baseline: List[str], 
        texts_target: List[str]
    ) -> Dict:
        """Détecte le glissement sémantique avec seuil statistique dynamique"""
        
        logger.info("DÉTECTION DU GLISSEMENT SÉMANTIQUE EN COURS...")
        
        # 1. Calcul des embeddings
        emb_base = self.compute_embeddings(texts_baseline)
        emb_target = self.compute_embeddings(texts_target)
        
        # 2. Calcul de la matrice de similarité (Target 2025 vs Baseline 2024)
        sim_matrix = cosine_similarity(emb_target, emb_base)
        
        # 3. Meilleure correspondance pour chaque phrase de 2025
        max_similarities = sim_matrix.max(axis=1)
        best_matches_idx = sim_matrix.argmax(axis=1)
        
        # 4. LOGIQUE DYNAMIQUE (Correction du 0%)
        # Au lieu d'un seuil fixe (0.75), on regarde l'écart à la moyenne.
        mean_sim = float(max_similarities.mean())
        std_sim = float(max_similarities.std())
        
        # Seuil = Moyenne - 1 Écart-type. 
        # On cible les phrases qui "décrochent" du reste du document.
        dynamic_threshold = mean_sim - std_sim
        
        drift_indices = np.where(max_similarities < dynamic_threshold)[0]
        drift_count = len(drift_indices)
        
        results = {
            'mean_similarity': mean_sim,
            'global_status': validate_similarity_score(mean_sim),
            'drifted_count': drift_count,
            'total_sentences_analysed': len(texts_target),
            'drifted_percentage': float((drift_count / len(texts_target)) * 100),
            'statistical_threshold': dynamic_threshold,
            'std_deviation': std_sim
        }
        
        # 5. Extraction des exemples (les similarités les plus basses)
        sorted_indices = np.argsort(max_similarities)
        results['drift_examples'] = [
            {
                'target_2025': texts_target[idx][:200],
                'best_match_2024_found': texts_baseline[best_matches_idx[idx]][:200],
                'similarity': float(max_similarities[idx])
            } for idx in sorted_indices[:5]
        ]
        
        return results

def run_full_analysis():
    """Fonction principale pour comparer les budgets 2024 et 2025"""
    analyzer = SemanticAnalyzer()
    
    # 1. Chargement des données traitées
    path_24 = config.RESULTS_DIR / "processed_2024.json"
    path_25 = config.RESULTS_DIR / "processed_2025.json"
    
    if not path_24.exists() or not path_25.exists():
        logger.error(f"Fichiers JSON introuvables dans {config.RESULTS_DIR}")
        return

    with open(path_24, 'r', encoding='utf-8') as f:
        data24 = json.load(f)
    with open(path_25, 'r', encoding='utf-8') as f:
        data25 = json.load(f)

    # Utilisation des sentences pour une analyse granulaire
    texts24 = data24.get('sentences', [])
    texts25 = data25.get('sentences', [])

    if not texts24 or not texts25:
        logger.error("Erreur : Les fichiers JSON ne contiennent pas de listes 'sentences'.")
        return

    logger.info(f"Analyse de glissement : 2025 ({len(texts25)} phrases) vs 2024 ({len(texts24)} phrases)")

    # 2. Exécution de l'analyse
    drift_results = analyzer.detect_semantic_drift(texts24, texts25)

    # 3. Sauvegarde
    save_results(drift_results, "semantic_drift_report")
    
    print("\n" + "="*60)
    print(f"RÉSULTAT DE L'AUDIT SÉMANTIQUE (MÉTHODE DYNAMIQUE)")
    print("-" * 60)
    print(f"Similarité moyenne  : {drift_results['mean_similarity']:.4f}")
    print(f"Écart-type          : {drift_results['std_deviation']:.4f}")
    print(f"Seuil de rupture    : {drift_results['statistical_threshold']:.4f}")
    print(f"Taux de changement  : {drift_results['drifted_percentage']:.2f}%")
    print(f"Nombre de ruptures  : {drift_results['drifted_count']} phrases")
    print("="*60)

if __name__ == "__main__":
    try:
        run_full_analysis()
    except Exception as e:
        logger.error(f"Erreur durant l'exécution : {e}", exc_info=True)