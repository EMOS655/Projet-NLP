"""
classification.py - Version CamemBERT de base
Utilise CamemBERT pour la classification des piliers SND30

Approche hybride:
1. CamemBERT pour les embeddings (représentation sémantique)
2. Classification par similarité cosinus avec les descriptions des piliers
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm

# Imports pour CamemBERT
from transformers import CamembertTokenizer, CamembertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    SND30_PILLARS,
    SND30_DESCRIPTIONS,
    SND30_KEYWORDS,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
    MODEL_NAME,
    CLASSIFICATION_CONFIDENCE_THRESHOLD,
    DEVICE,
    MAX_LENGTH,
    RANDOM_SEED
)

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NLP_Project')


class CamemBERTClassifier:
    """
    Classificateur utilisant CamemBERT de base pour la classification des piliers SND30
    
    Méthode:
    1. Encode le texte avec CamemBERT
    2. Encode les descriptions des piliers SND30
    3. Calcule la similarité cosinus
    4. Retourne le pilier le plus similaire
    """
    
    def __init__(self, model_name: str = "camembert-base", device: str = None):
        """
        Initialise le classificateur CamemBERT
        
        Args:
            model_name: Nom du modèle CamemBERT (par défaut: camembert-base)
            device: Device à utiliser (cuda/cpu)
        """
        logger.info(f"Initialisation du classificateur CamemBERT: {model_name}")
        
        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Device utilisé: {self.device}")
        
        # Chargement du modèle et tokenizer
        logger.info(f"Chargement de CamemBERT: {model_name}")
        self.tokenizer = CamembertTokenizer.from_pretrained(model_name)
        self.model = CamembertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Préparation des embeddings des piliers
        logger.info("Calcul des embeddings des piliers SND30")
        self.pillar_embeddings = self._compute_pillar_embeddings()
        
        logger.info("✓ Classificateur CamemBERT prêt")
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Obtient l'embedding CamemBERT d'un texte
        
        Args:
            text: Texte à encoder
            
        Returns:
            Vecteur embedding (numpy array)
        """
        # Tokenization
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Utilise le [CLS] token (premier token) comme représentation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]
    
    def _compute_pillar_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Calcule les embeddings pour chaque pilier SND30
        
        Returns:
            Dictionnaire {pilier: embedding}
        """
        pillar_embeddings = {}
        
        for pillar in SND30_PILLARS:
            # Combine description et keywords pour une meilleure représentation
            description = SND30_DESCRIPTIONS[pillar]
            keywords = ", ".join(SND30_KEYWORDS[pillar][:10])  # Top 10 keywords
            
            # Texte complet pour le pilier
            pillar_text = f"{pillar}. {description}. Mots-clés: {keywords}"
            
            # Calcul de l'embedding
            embedding = self._get_embedding(pillar_text)
            pillar_embeddings[pillar] = embedding
            
            logger.info(f"  - Embedding calculé pour: {pillar}")
        
        return pillar_embeddings
    
    def classify(self, text: str, return_scores: bool = False) -> Dict:
        """
        Classifie un texte dans un pilier SND30
        
        Args:
            text: Texte à classifier
            return_scores: Si True, retourne les scores pour tous les piliers
            
        Returns:
            Dictionnaire avec le pilier prédit et le score de confiance
        """
        # Embedding du texte
        text_embedding = self._get_embedding(text)
        
        # Calcul des similarités avec chaque pilier
        similarities = {}
        for pillar, pillar_emb in self.pillar_embeddings.items():
            # Similarité cosinus
            sim = cosine_similarity(
                text_embedding.reshape(1, -1),
                pillar_emb.reshape(1, -1)
            )[0][0]
            similarities[pillar] = float(sim)
        
        # Pilier avec la plus haute similarité
        predicted_pillar = max(similarities, key=similarities.get)
        confidence = similarities[predicted_pillar]
        
        # NOUVEAU: Vérifier si la différence est significative
        # Si tous les scores sont trop proches, classifier comme "Autre"
        sorted_scores = sorted(similarities.values(), reverse=True)
        if len(sorted_scores) >= 2:
            top_score = sorted_scores[0]
            second_score = sorted_scores[1]
            score_difference = top_score - second_score
            
            # Si la différence est trop faible (< 0.05), c'est probablement "Autre"
            if score_difference < 0.05 and predicted_pillar != "Autre":
                # Vérifier si "Autre" a un score compétitif
                if "Autre" in similarities and similarities["Autre"] > second_score - 0.02:
                    predicted_pillar = "Autre"
                    confidence = similarities["Autre"]
        
        # Résultat
        result = {
            "predicted_pillar": predicted_pillar,
            "confidence": confidence,
            "method": "camembert-similarity"
        }
        
        if return_scores:
            result["all_scores"] = similarities
        
        return result
    
    def classify_batch(self, texts: List[str], show_progress: bool = True) -> List[Dict]:
        """
        Classifie un batch de textes
        
        Args:
            texts: Liste de textes à classifier
            show_progress: Afficher la barre de progression
            
        Returns:
            Liste de résultats de classification
        """
        results = []
        
        iterator = tqdm(texts, desc="Classification") if show_progress else texts
        
        for text in iterator:
            result = self.classify(text, return_scores=True)
            results.append(result)
        
        return results


def classify_budget_items(
    year: int,
    item_type: str = "articles",
    confidence_threshold: float = None,
    save_results: bool = True
) -> Dict:
    """
    Classifie les éléments budgétaires d'une année dans les piliers SND30
    
    Args:
        year: Année du document (2024, 2025, etc.)
        item_type: Type d'éléments ("articles" ou "sections")
        confidence_threshold: Seuil de confiance minimum
        save_results: Sauvegarder les résultats
        
    Returns:
        Résultats de classification
    """
    logger.info("=" * 80)
    logger.info("CLASSIFICATION DES ÉLÉMENTS BUDGÉTAIRES DANS LES PILIERS SND30")
    logger.info("=" * 80)
    
    # Chargement des données
    data_file = PROCESSED_DATA_DIR / f"processed_{year}.json"
    logger.info(f"Document : {data_file.name}")
    logger.info(f"Type d'éléments : {item_type}")
    
    if not data_file.exists():
        logger.error(f"Fichier non trouvé : {data_file}")
        return None
    
    with open(data_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    items = data.get(item_type, [])
    logger.info(f"Nombre d'éléments à classifier : {len(items)}")
    
    # Initialisation du classificateur
    logger.info(f"Chargement du modèle CamemBERT: {MODEL_NAME}")
    if confidence_threshold is None:
        confidence_threshold = CLASSIFICATION_CONFIDENCE_THRESHOLD
    logger.info(f"Seuil de confiance : {confidence_threshold}")
    
    try:
        classifier = CamemBERTClassifier(model_name=MODEL_NAME)
    except Exception as e:
        logger.error(f"Erreur lors du chargement du modèle : {e}")
        raise
    
    # Préparation des textes
    texts = []
    for item in items:
        # Gestion des différents formats de données
        if isinstance(item, str):
            # Si l'item est déjà une string
            text = item
        elif isinstance(item, dict):
            # Si l'item est un dictionnaire
            if item_type == "articles":
                text = f"Article {item.get('numero', 'N/A')}: {item.get('titre', '')} {item.get('contenu', '')}"
            else:
                text = f"{item.get('titre', '')} {item.get('contenu', '')}"
        else:
            # Fallback
            text = str(item)
        
        texts.append(text)
    
    # Classification
    logger.info(f"\n🔄 Classification en cours avec CamemBERT...")
    classifications = classifier.classify_batch(texts)
    
    # Ajout des classifications aux items
    classified_items = []
    for i, (item, classification) in enumerate(zip(items, classifications)):
        if isinstance(item, dict):
            # Si l'item est un dictionnaire, on ajoute la classification
            item_with_classification = {
                **item,
                "pilier_snd30": classification["predicted_pillar"],
                "confiance": classification["confidence"],
                "methode_classification": classification["method"],
                "scores_piliers": classification.get("all_scores", {})
            }
        else:
            # Si l'item est une string, on crée un nouveau dictionnaire
            item_with_classification = {
                "texte": str(item),
                "index": i,
                "pilier_snd30": classification["predicted_pillar"],
                "confiance": classification["confidence"],
                "methode_classification": classification["method"],
                "scores_piliers": classification.get("all_scores", {})
            }
        classified_items.append(item_with_classification)
    
    # Statistiques
    pillar_counts = {}
    for item in classified_items:
        pillar = item["pilier_snd30"]
        pillar_counts[pillar] = pillar_counts.get(pillar, 0) + 1
    
    # Calcul des statistiques
    results = {
        "year": year,
        "item_type": item_type,
        "total_items": len(classified_items),
        "classification_method": "camembert-similarity",
        "model_name": MODEL_NAME,
        "confidence_threshold": confidence_threshold,
        "pillar_distribution": pillar_counts,
        "classified_items": classified_items,
        "statistics": {
            "average_confidence": np.mean([item["confiance"] for item in classified_items]),
            "min_confidence": np.min([item["confiance"] for item in classified_items]),
            "max_confidence": np.max([item["confiance"] for item in classified_items])
        }
    }
    
    # Affichage des résultats
    logger.info("\n" + "=" * 80)
    logger.info("RÉSULTATS DE LA CLASSIFICATION")
    logger.info("=" * 80)
    logger.info(f"\n📊 Distribution par pilier SND30:")
    for pillar in SND30_PILLARS:
        count = pillar_counts.get(pillar, 0)
        percentage = (count / len(classified_items)) * 100
        logger.info(f"  • {pillar}: {count} ({percentage:.1f}%)")
    
    logger.info(f"\n📈 Statistiques de confiance:")
    logger.info(f"  • Moyenne: {results['statistics']['average_confidence']:.3f}")
    logger.info(f"  • Min: {results['statistics']['min_confidence']:.3f}")
    logger.info(f"  • Max: {results['statistics']['max_confidence']:.3f}")
    
    # Sauvegarde
    if save_results:
        output_file = RESULTS_DIR / f"classification_camembert_{year}_{item_type}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✓ Résultats sauvegardés : {output_file}")
    
    return results


def compare_classifications_2024_2025(save_outputs: bool = True) -> Dict:
    """
    Compare les classifications entre 2024 et 2025
    
    Args:
        save_outputs: Sauvegarder les résultats de comparaison
        
    Returns:
        Dictionnaire avec les résultats de comparaison
    """
    logger.info("=" * 80)
    logger.info("COMPARAISON DES CLASSIFICATIONS 2024 vs 2025")
    logger.info("=" * 80)
    
    # Classification 2024
    logger.info("\n📊 Classification des articles 2024...")
    results_2024 = classify_budget_items(
        year=2024,
        item_type="articles",
        save_results=save_outputs
    )
    
    # Classification 2025
    logger.info("\n📊 Classification des articles 2025...")
    results_2025 = classify_budget_items(
        year=2025,
        item_type="articles",
        save_results=save_outputs
    )
    
    if results_2024 is None or results_2025 is None:
        logger.error("❌ Erreur lors de la classification")
        return None
    
    # Calcul des différences
    comparison = {
        "year_2024": results_2024["pillar_distribution"],
        "year_2025": results_2025["pillar_distribution"],
        "changes": {}
    }
    
    logger.info("\n" + "=" * 80)
    logger.info("ÉVOLUTION 2024 → 2025")
    logger.info("=" * 80)
    
    for pillar in SND30_PILLARS:
        count_2024 = results_2024["pillar_distribution"].get(pillar, 0)
        count_2025 = results_2025["pillar_distribution"].get(pillar, 0)
        
        pct_2024 = (count_2024 / results_2024["total_items"]) * 100
        pct_2025 = (count_2025 / results_2025["total_items"]) * 100
        
        change = count_2025 - count_2024
        pct_change = pct_2025 - pct_2024
        
        comparison["changes"][pillar] = {
            "count_2024": count_2024,
            "count_2025": count_2025,
            "absolute_change": change,
            "percentage_change": pct_change
        }
        
        symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
        logger.info(f"\n{symbol} {pillar}:")
        logger.info(f"  2024: {count_2024} ({pct_2024:.1f}%)")
        logger.info(f"  2025: {count_2025} ({pct_2025:.1f}%)")
        logger.info(f"  Δ: {change:+d} ({pct_change:+.1f}%)")
    
    # Sauvegarde de la comparaison
    if save_outputs:
        comparison_file = RESULTS_DIR / "comparison_2024_2025_camembert.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        logger.info(f"\n✓ Comparaison sauvegardée : {comparison_file}")
    
    return comparison


if __name__ == "__main__":
    logger.info("Logging system initialized")
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DE LA CLASSIFICATION CAMEMBERT")
    logger.info("=" * 80)
    
    try:
        # Comparaison 2024 vs 2025
        comparison = compare_classifications_2024_2025(save_outputs=True)
        
        if comparison:
            logger.info("\n" + "=" * 80)
            logger.info("✓ CLASSIFICATION TERMINÉE AVEC SUCCÈS")
            logger.info("=" * 80)
        else:
            logger.error("\n✗ La classification a échoué. Consultez les logs pour plus de détails.")
    
    except Exception as e:
        logger.error(f"\n❌ Erreur fatale : {e}", exc_info=True)
        print("\n✗ La classification a échoué. Consultez les logs pour plus de détails.")