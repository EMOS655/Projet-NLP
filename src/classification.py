import os
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from tqdm import tqdm

import torch
from transformers import CamembertTokenizer, CamembertModel
from sklearn.metrics.pairwise import cosine_similarity

# Configuration du chemin projet pour les imports locaux
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from src.utils import setup_logging, save_results, Timer

# OPTIMISATION CPU : Utilise tous les coeurs disponibles
torch.set_num_threads(os.cpu_count())

logger = setup_logging()

class CamemBERTHybridClassifier:
    """Classificateur hybride optimisé pour CPU"""
    
    def __init__(self, model_name: str = None, semantic_weight: float = 0.6, keyword_weight: float = 0.4):
        self.model_name = model_name or "camembert-base"
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.device = torch.device("cpu")
        
        logger.info(f"Initialisation CamemBERT sur CPU ({self.semantic_weight*100}% sémantique)")
        
        # Chargement local ou distant
        self.tokenizer = CamembertTokenizer.from_pretrained(self.model_name)
        self.model = CamembertModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Pré-calcul des embeddings des piliers pour gagner du temps
        logger.info("Calcul des embeddings de référence pour les piliers SND30...")
        self.pillar_embeddings = self._compute_pillar_embeddings()
        logger.info("✓ Classificateur prêt")

    def _get_embedding(self, text: str) -> np.ndarray:
        """Calcule l'embedding avec Mean Pooling"""
        if not text or len(text.strip()) == 0:
            return np.zeros((768,))

        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mean Pooling (Moyenne des vecteurs pour une meilleure représentation)
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).cpu().numpy()
        
        return embedding[0]

    def _compute_pillar_embeddings(self) -> Dict[str, np.ndarray]:
        embeddings = {}
        for pillar in config.SND30_PILLARS:
            desc = config.SND30_DESCRIPTIONS.get(pillar, "")
            kws = ", ".join(config.SND30_KEYWORDS.get(pillar, [])[:15])
            full_text = f"{pillar}: {desc}. {kws}"
            embeddings[pillar] = self._get_embedding(full_text)
        return embeddings

    def _compute_keyword_scores(self, text: str) -> Dict[str, float]:
        text_lower = text.lower()
        scores = {}
        for pillar in config.SND30_PILLARS:
            keywords = config.SND30_KEYWORDS.get(pillar, [])
            count = sum(1 for kw in keywords if kw.lower() in text_lower)
            scores[pillar] = count / len(keywords) if keywords else 0.0
            
        total = sum(scores.values())
        if total > 0:
            return {k: v/total for k, v in scores.items()}
        return {k: 1.0/len(config.SND30_PILLARS) for k in config.SND30_PILLARS}

    def classify(self, text: str) -> Dict:
        # 1. Score Sémantique
        text_emb = self._get_embedding(text)
        sem_scores = {}
        for pillar, p_emb in self.pillar_embeddings.items():
            sim = cosine_similarity(text_emb.reshape(1, -1), p_emb.reshape(1, -1))[0][0]
            sem_scores[pillar] = max(0, float(sim))
            
        total_sem = sum(sem_scores.values())
        if total_sem > 0:
            sem_scores = {k: v/total_sem for k, v in sem_scores.items()}

        # 2. Score Keywords
        kw_scores = self._compute_keyword_scores(text)

        # 3. Fusion Hybride
        hybrid_scores = {}
        for p in config.SND30_PILLARS:
            hybrid_scores[p] = (self.semantic_weight * sem_scores[p]) + (self.keyword_weight * kw_scores[p])

        best_pillar = max(hybrid_scores, key=hybrid_scores.get)
        return {
            "predicted_pillar": best_pillar,
            "confidence": hybrid_scores[best_pillar],
            "all_scores": hybrid_scores
        }

def run_comparison():
    """Fonction principale de classification 2024 vs 2025"""
    classifier = CamemBERTHybridClassifier(model_name=config.MODEL_NAME)
    
    final_comparison = {}
    
    # Correction de l'accès aux fichiers (on regarde dans data/results suite à ton log)
    input_dir = config.RESULTS_DIR 
    
    for year in [2024, 2025]:
        filename = f"processed_{year}.json"
        path = input_dir / filename
        
        if not path.exists():
            logger.error(f"Fichier introuvable: {path}")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # On récupère les articles (liste)
        articles = data.get('articles', [])
        logger.info(f"Analyse Année {year} : {len(articles)} articles trouvés.")
        
        classified_results = []
        pillar_dist = {p: 0 for p in config.SND30_PILLARS}
        
        for art in tqdm(articles, desc=f"Classif {year}"):
            # --- CORRECTION DE L'ERREUR ATTRIBUTERROR ---
            if isinstance(art, dict):
                # Si c'est un dictionnaire, on extrait proprement les champs
                t = art.get('titre', '') if art.get('titre') else ""
                c = art.get('contenu', '') if art.get('contenu') else ""
                txt = f"{t} {c}".strip()
            elif isinstance(art, str):
                # Si c'veut dire que l'article est juste une chaîne de caractères
                txt = art.strip()
            else:
                continue

            if not txt:
                continue
                
            prediction = classifier.classify(txt)
            
            # Reconstruction d'un objet propre pour le résultat
            res_item = {
                "article_original": art if isinstance(art, str) else art.get('numero', 'N/A'),
                "pilier_snd30": prediction["predicted_pillar"],
                "confiance": float(prediction["confidence"])
            }
            
            classified_results.append(res_item)
            pillar_dist[prediction["predicted_pillar"]] += 1
            
        # Sauvegarde des détails par année
        save_results(classified_results, f"classification_details_{year}")
        final_comparison[year] = pillar_dist

    # Sauvegarde du rapport final
    if final_comparison:
        save_results(final_comparison, "comparison_piliers_2024_2025")
        logger.info("🚀 Classification et comparaison terminées !")
        
        # Affichage rapide dans la console
        print("\n--- RÉSUMÉ DES PILIERS ---")
        for yr, dist in final_comparison.items():
            print(f"\nAnnée {yr}:")
            for p, count in dist.items():
                print(f"  {p}: {count}")

if __name__ == "__main__":
    try:
        with Timer("Classification complète"):
            run_comparison()
    except Exception as e:
        logger.error(f"Erreur fatale : {e}", exc_info=True)