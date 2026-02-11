import sys
from pathlib import Path

# Ajouter le dossier parent au PYTHONPATH
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import numpy as np
from typing import List, Dict, Tuple
import json
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("⚠️ sentence-transformers non installé")
    print("Installez avec : pip install sentence-transformers")

from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

import config
from src.utils import setup_logging, save_results, Timer

# Setup logging
logger = setup_logging()


class SemanticAnalyzer:
    """Classe pour l'analyse sémantique avec embeddings"""
    
    def __init__(self, model_name: str = None):
        """
        Initialise l'analyseur sémantique
        
        Args:
            model_name: Nom du modèle Sentence-BERT à utiliser
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers est requis. "
                "Installez-le avec : pip install sentence-transformers"
            )
        
        self.model_name = model_name or config.SENTENCE_BERT_MODEL
        logger.info(f"Chargement du modèle Sentence-BERT : {self.model_name}")
        logger.info("⏳ Premier chargement : téléchargement du modèle (~400 MB)...")
        
        with Timer(f"Chargement du modèle {self.model_name}"):
            self.model = SentenceTransformer(self.model_name)
        
        logger.info(f"✓ Modèle chargé avec succès")
        logger.info(f"✓ Dimension des embeddings : {self.model.get_sentence_embedding_dimension()}")
    
    def compute_embeddings(self, texts: List[str], batch_size: int = None) -> np.ndarray:
        """
        Calcule les embeddings pour une liste de textes
        
        Args:
            texts: Liste de textes à encoder
            batch_size: Taille des batches pour l'encodage
            
        Returns:
            Array numpy des embeddings
        """
        if not texts:
            logger.warning("Liste de textes vide")
            return np.array([])
        
        batch_size = batch_size or config.BATCH_SIZE
        
        logger.info(f"Calcul des embeddings pour {len(texts)} textes...")
        
        with Timer(f"Encodage de {len(texts)} textes"):
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalisation pour meilleure similarité cosinus
            )
        
        logger.info(f"✓ Embeddings calculés : shape {embeddings.shape}")
        return embeddings
    
    def calculate_similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Calcule la similarité cosinus entre deux ensembles d'embeddings
        
        Args:
            embeddings1: Premier ensemble d'embeddings
            embeddings2: Deuxième ensemble d'embeddings
            
        Returns:
            Matrice de similarité
        """
        logger.info(f"Calcul de similarité : {embeddings1.shape} vs {embeddings2.shape}")
        
        similarity_matrix = cosine_similarity(embeddings1, embeddings2)
        
        logger.info(f"✓ Matrice de similarité calculée : {similarity_matrix.shape}")
        logger.info(f"  • Similarité moyenne : {similarity_matrix.mean():.4f}")
        logger.info(f"  • Similarité min : {similarity_matrix.min():.4f}")
        logger.info(f"  • Similarité max : {similarity_matrix.max():.4f}")
        
        return similarity_matrix
    
    def find_most_similar(
        self,
        query_text: str,
        corpus_texts: List[str],
        corpus_embeddings: np.ndarray = None,
        top_k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Trouve les textes les plus similaires à une requête
        
        Args:
            query_text: Texte de requête
            corpus_texts: Liste de textes du corpus
            corpus_embeddings: Embeddings pré-calculés du corpus (optionnel)
            top_k: Nombre de résultats à retourner
            
        Returns:
            Liste de (index, texte, score de similarité)
        """
        # Encoder la requête
        query_embedding = self.model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        
        # Calculer ou utiliser les embeddings du corpus
        if corpus_embeddings is None:
            corpus_embeddings = self.compute_embeddings(corpus_texts)
        
        # Calculer similarités
        similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
        
        # Trouver les top_k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (int(idx), corpus_texts[idx], float(similarities[idx]))
            for idx in top_indices
        ]
        
        return results
    
    def detect_semantic_drift(
        self,
        texts_baseline: List[str],
        texts_target: List[str],
        threshold: float = None
    ) -> Dict:
        """
        Détecte le glissement sémantique entre deux corpus
        
        Args:
            texts_baseline: Textes de référence (baseline)
            texts_target: Textes à comparer (target)
            threshold: Seuil de similarité pour détecter un drift
            
        Returns:
            Dictionnaire avec statistiques de drift
        """
        threshold = threshold or config.DRIFT_THRESHOLD
        
        logger.info("=" * 80)
        logger.info("DÉTECTION DU GLISSEMENT SÉMANTIQUE")
        logger.info("=" * 80)
        
        # Calculer les embeddings
        embeddings_baseline = self.compute_embeddings(texts_baseline)
        embeddings_target = self.compute_embeddings(texts_target)
        
        # Calculer la similarité
        similarity_matrix = self.calculate_similarity(embeddings_baseline, embeddings_target)
        
        # Pour chaque texte baseline, trouver la meilleure correspondance dans target
        max_similarities = similarity_matrix.max(axis=1)
        best_matches = similarity_matrix.argmax(axis=1)
        
        # Statistiques
        drift_stats = {
            'mean_similarity': float(max_similarities.mean()),
            'median_similarity': float(np.median(max_similarities)),
            'min_similarity': float(max_similarities.min()),
            'max_similarity': float(max_similarities.max()),
            'std_similarity': float(max_similarities.std()),
            'threshold': threshold,
            'num_baseline': len(texts_baseline),
            'num_target': len(texts_target),
            'drifted_count': int((max_similarities < threshold).sum()),
            'drifted_percentage': float((max_similarities < threshold).mean() * 100),
            'stable_count': int((max_similarities >= config.SIMILARITY_THRESHOLD_MEDIUM).sum()),
            'stable_percentage': float((max_similarities >= config.SIMILARITY_THRESHOLD_MEDIUM).mean() * 100)
        }
        
        # Identifier les textes avec drift
        drifted_indices = np.where(max_similarities < threshold)[0]
        drift_stats['drifted_texts'] = [
            {
                'index': int(idx),
                'baseline_text': texts_baseline[idx][:200] + "..." if len(texts_baseline[idx]) > 200 else texts_baseline[idx],
                'best_match_text': texts_target[best_matches[idx]][:200] + "..." if len(texts_target[best_matches[idx]]) > 200 else texts_target[best_matches[idx]],
                'max_similarity': float(max_similarities[idx])
            }
            for idx in drifted_indices[:10]  # Limiter à 10 exemples
        ]
        
        logger.info(f"✓ Similarité moyenne : {drift_stats['mean_similarity']:.4f}")
        logger.info(f"✓ Textes avec drift significatif : {drift_stats['drifted_count']} ({drift_stats['drifted_percentage']:.2f}%)")
        logger.info(f"✓ Textes stables : {drift_stats['stable_count']} ({drift_stats['stable_percentage']:.2f}%)")
        
        return drift_stats


def analyze_documents_similarity(
    doc1_path: Path = None,
    doc2_path: Path = None,
    save_outputs: bool = True
) -> Dict:
    """
    Analyse la similarité entre deux documents
    
    Args:
        doc1_path: Chemin vers le premier document JSON
        doc2_path: Chemin vers le deuxième document JSON
        save_outputs: Si True, sauvegarde les résultats
        
    Returns:
        Dictionnaire avec les résultats d'analyse
    """
    logger.info("=" * 80)
    logger.info("ANALYSE DE SIMILARITÉ ENTRE DOCUMENTS")
    logger.info("=" * 80)
    
    # Charger les documents
    doc1_path = doc1_path or config.RESULTS_DIR / "processed_2024.json"
    doc2_path = doc2_path or config.RESULTS_DIR / "processed_2025.json"
    
    logger.info(f"Document 1 : {doc1_path.name}")
    logger.info(f"Document 2 : {doc2_path.name}")
    
    with open(doc1_path, 'r', encoding='utf-8') as f:
        doc1 = json.load(f)
    
    with open(doc2_path, 'r', encoding='utf-8') as f:
        doc2 = json.load(f)
    
    # Initialiser l'analyseur
    analyzer = SemanticAnalyzer()
    
    # Analyser au niveau des phrases
    logger.info("\n" + "-" * 80)
    logger.info("ANALYSE AU NIVEAU DES PHRASES")
    logger.info("-" * 80)
    
    sentences1 = doc1['sentences']
    sentences2 = doc2['sentences']
    
    logger.info(f"Document 1 : {len(sentences1)} phrases")
    logger.info(f"Document 2 : {len(sentences2)} phrases")
    
    sentence_drift = analyzer.detect_semantic_drift(sentences1, sentences2)
    
    # Analyser au niveau des paragraphes
    logger.info("\n" + "-" * 80)
    logger.info("ANALYSE AU NIVEAU DES PARAGRAPHES")
    logger.info("-" * 80)
    
    paragraphs1 = doc1['paragraphs']
    paragraphs2 = doc2['paragraphs']
    
    logger.info(f"Document 1 : {len(paragraphs1)} paragraphes")
    logger.info(f"Document 2 : {len(paragraphs2)} paragraphes")
    
    paragraph_drift = analyzer.detect_semantic_drift(paragraphs1, paragraphs2)
    
    # Analyser au niveau des articles (si disponibles)
    article_drift = None
    if 'articles' in doc1 and 'articles' in doc2:
        logger.info("\n" + "-" * 80)
        logger.info("ANALYSE AU NIVEAU DES ARTICLES")
        logger.info("-" * 80)
        
        articles1 = list(doc1['articles'].values())
        articles2 = list(doc2['articles'].values())
        
        logger.info(f"Document 1 : {len(articles1)} articles")
        logger.info(f"Document 2 : {len(articles2)} articles")
        
        if articles1 and articles2:
            article_drift = analyzer.detect_semantic_drift(articles1, articles2)
    
    # Résultats globaux
    results = {
        'documents': {
            'doc1': doc1_path.name,
            'doc2': doc2_path.name
        },
        'sentence_analysis': sentence_drift,
        'paragraph_analysis': paragraph_drift,
        'article_analysis': article_drift,
        'global_assessment': {
            'overall_similarity': (
                sentence_drift['mean_similarity'] * 0.5 +
                paragraph_drift['mean_similarity'] * 0.5
            ),
            'semantic_drift_detected': (
                sentence_drift['mean_similarity'] < config.SIMILARITY_THRESHOLD_MEDIUM or
                paragraph_drift['mean_similarity'] < config.SIMILARITY_THRESHOLD_MEDIUM
            ),
            'drift_severity': 'high' if sentence_drift['mean_similarity'] < 0.5 else 
                            'medium' if sentence_drift['mean_similarity'] < 0.7 else 'low'
        }
    }
    
    # Sauvegarder
    if save_outputs:
        save_results(results, "similarity_analysis_2024_2025", format='json')
        logger.info(f"✓ Résultats sauvegardés dans : {config.RESULTS_DIR}")
    
    return results


def compare_with_snd30(
    doc_path: Path = None,
    snd30_path: Path = None,
    save_outputs: bool = True
) -> Dict:
    """
    Compare un document avec la SND30
    
    Args:
        doc_path: Chemin vers le document à analyser
        snd30_path: Chemin vers le document SND30
        save_outputs: Si True, sauvegarde les résultats
        
    Returns:
        Dictionnaire avec les résultats de comparaison
    """
    logger.info("=" * 80)
    logger.info("COMPARAISON AVEC LA SND30")
    logger.info("=" * 80)
    
    # Charger les documents
    doc_path = doc_path or config.RESULTS_DIR / "processed_2025.json"
    snd30_path = snd30_path or config.RESULTS_DIR / "processed_SND30.json"
    
    logger.info(f"Document à analyser : {doc_path.name}")
    logger.info(f"Document SND30 : {snd30_path.name}")
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    
    with open(snd30_path, 'r', encoding='utf-8') as f:
        snd30 = json.load(f)
    
    # Initialiser l'analyseur
    analyzer = SemanticAnalyzer()
    
    # Analyser l'alignement global
    logger.info("\nAnalyse de l'alignement avec la SND30...")
    
    doc_sentences = doc['sentences']
    snd30_sentences = snd30['sentences']
    
    # Calculer embeddings
    doc_embeddings = analyzer.compute_embeddings(doc_sentences)
    snd30_embeddings = analyzer.compute_embeddings(snd30_sentences)
    
    # Calculer similarité
    similarity_matrix = analyzer.calculate_similarity(doc_embeddings, snd30_embeddings)
    
    # Pour chaque phrase du document, trouver la meilleure correspondance dans SND30
    max_similarities = similarity_matrix.max(axis=1)
    best_matches = similarity_matrix.argmax(axis=1)
    
    # Statistiques d'alignement
    alignment_stats = {
        'mean_alignment': float(max_similarities.mean()),
        'median_alignment': float(np.median(max_similarities)),
        'highly_aligned_count': int((max_similarities >= config.SIMILARITY_THRESHOLD_HIGH).sum()),
        'highly_aligned_percentage': float((max_similarities >= config.SIMILARITY_THRESHOLD_HIGH).mean() * 100),
        'moderately_aligned_count': int((
            (max_similarities >= config.SIMILARITY_THRESHOLD_MEDIUM) &
            (max_similarities < config.SIMILARITY_THRESHOLD_HIGH)
        ).sum()),
        'poorly_aligned_count': int((max_similarities < config.SIMILARITY_THRESHOLD_MEDIUM).sum()),
        'poorly_aligned_percentage': float((max_similarities < config.SIMILARITY_THRESHOLD_MEDIUM).mean() * 100),
        'alignment_quality': 'excellent' if max_similarities.mean() >= 0.75 else
                           'good' if max_similarities.mean() >= 0.60 else
                           'moderate' if max_similarities.mean() >= 0.45 else 'poor'
    }
    
    # Top alignements
    top_aligned_indices = np.argsort(max_similarities)[::-1][:10]
    alignment_stats['top_alignments'] = [
        {
            'doc_sentence': doc_sentences[idx][:200] + "..." if len(doc_sentences[idx]) > 200 else doc_sentences[idx],
            'snd30_sentence': snd30_sentences[best_matches[idx]][:200] + "..." if len(snd30_sentences[best_matches[idx]]) > 200 else snd30_sentences[best_matches[idx]],
            'similarity': float(max_similarities[idx])
        }
        for idx in top_aligned_indices
    ]
    
    logger.info(f"✓ Alignement moyen avec SND30 : {alignment_stats['mean_alignment']:.4f}")
    logger.info(f"✓ Qualité de l'alignement : {alignment_stats['alignment_quality'].upper()}")
    logger.info(f"✓ Phrases hautement alignées : {alignment_stats['highly_aligned_count']} ({alignment_stats['highly_aligned_percentage']:.2f}%)")
    logger.info(f"✓ Phrases faiblement alignées : {alignment_stats['poorly_aligned_count']} ({alignment_stats['poorly_aligned_percentage']:.2f}%)")
    
    results = {
        'document': doc_path.name,
        'alignment_with_snd30': alignment_stats
    }
    
    # Sauvegarder
    if save_outputs:
        filename = f"snd30_alignment_{doc_path.stem}"
        save_results(results, filename, format='json')
        logger.info(f"✓ Résultats sauvegardés")
    
    return results


if __name__ == "__main__":
    logger.info("=" * 80)
    logger.info("DÉMARRAGE DE L'ANALYSE SÉMANTIQUE")
    logger.info("=" * 80)
    
    try:
        # Analyse 1 : Similarité 2024 vs 2025
        print("\n🔍 Analyse de la similarité 2024 vs 2025...")
        similarity_results = analyze_documents_similarity(save_outputs=True)
        
        print("\n📊 Résultats :")
        print(f"  • Similarité globale : {similarity_results['global_assessment']['overall_similarity']:.4f}")
        print(f"  • Glissement sémantique : {'OUI ⚠️' if similarity_results['global_assessment']['semantic_drift_detected'] else 'NON ✓'}")
        print(f"  • Sévérité du drift : {similarity_results['global_assessment']['drift_severity'].upper()}")
        
        # Analyse 2 : Comparaison avec SND30 (2024)
        print("\n🎯 Comparaison Loi 2024 avec SND30...")
        snd30_2024 = compare_with_snd30(
            doc_path=config.PROCESSED_DATA_DIR / "processed_2024.json",
            save_outputs=True
        )
        
        print(f"  • Alignement moyen : {snd30_2024['alignment_with_snd30']['mean_alignment']:.4f}")
        print(f"  • Qualité : {snd30_2024['alignment_with_snd30']['alignment_quality'].upper()}")
        
        # Analyse 3 : Comparaison avec SND30 (2025)
        print("\n🎯 Comparaison Loi 2025 avec SND30...")
        snd30_2025 = compare_with_snd30(
            doc_path=config.PROCESSED_DATA_DIR / "processed_2025.json",
            save_outputs=True
        )
        
        print(f"  • Alignement moyen : {snd30_2025['alignment_with_snd30']['mean_alignment']:.4f}")
        print(f"  • Qualité : {snd30_2025['alignment_with_snd30']['alignment_quality'].upper()}")
        
        # Résumé final
        print("\n" + "=" * 80)
        print("✅ ANALYSE TERMINÉE AVEC SUCCÈS")
        print("=" * 80)
        print(f"\n📁 Fichiers générés dans : {config.RESULTS_DIR}")
        print("\nProchaine étape : Classification zero-shot dans les piliers SND30")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'analyse : {e}")
        import traceback
        traceback.print_exc()
        print("\n✗ L'analyse a échoué. Consultez les logs pour plus de détails.")