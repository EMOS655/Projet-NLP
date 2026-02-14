"""
Configuration file for NLP Finance Law Analysis Project
ISSEA - Yaoundé
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

# Base directory
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"
LABELED_DATA_DIR = DATA_DIR / "labeled"

# Model directories
MODELS_DIR = BASE_DIR / "models"
PRETRAINED_DIR = MODELS_DIR / "pretrained"
FINETUNED_DIR = MODELS_DIR / "finetuned"
CHECKPOINT_DIR = FINETUNED_DIR / "checkpoints"

# Output directories
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Notebook directory
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Logs directory
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, LABELED_DATA_DIR,
                  PRETRAINED_DIR, FINETUNED_DIR, CHECKPOINT_DIR, FIGURES_DIR, 
                  TABLES_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# SND30 (Stratégie Nationale de Développement 2020-2030)
# ============================================================================

SND30_PILLARS = [
    "Transformation structurelle",
    "Capital humain",
    "Gouvernance",
    "Développement régional",
    "Autre"  # Articles qui ne correspondent à aucun pilier SND30
]

# Detailed descriptions for zero-shot and sentence-BERT classification
SND30_DESCRIPTIONS = {
    "Transformation structurelle": (
        "Focus sur l'économie productive et l'industrialisation. Projets de grande envergure pour changer la structure économique : "
        "agro-industrie, manufactures, secteur extractif (mines, pétrole, gaz) et énergie (barrages, électricité). "
        "Infrastructures lourdes : autoroutes, ports autonomes (Kribi, Douala), chemins de fer. "
        "Innovation technologique, économie numérique et compétitivité des filières bois, cacao et coton."
    ),
    "Capital humain": (
        "Investissement dans la population et le bien-être social. Système éducatif complet (primaire, secondaire, supérieur, technique). "
        "Santé publique : Couverture Santé Universelle (CSU), hôpitaux généraux, lutte contre les maladies, bourses d'études. "
        "Inclusion et filets sociaux : protection des démunis, pensions de retraite, emploi des jeunes, formation professionnelle "
        "et entrepreneuriat social."
    ),
    "Gouvernance": (
        "Fonctionnement régalien, sécurité et réforme de l'État. Institutions nationales, justice, tribunaux et État de droit. "
        "Défense nationale : gendarmerie, armée, maintien de l'ordre. Transparence et lutte contre la corruption (audits, CONAC). "
        "Modernisation de l'administration publique, diplomatie et gestion efficace des marchés publics."
    ),
    "Développement régional": (
        "Priorité aux territoires et à la décentralisation. Appui direct aux Collectivités Territoriales Décentralisées (CTD). "
        "Aménagement local : désenclavement rural, pistes de collecte, ponts communaux, électrification villageoise. "
        "Services de proximité : accès à l'eau potable (forages), marchés locaux, assainissement urbain et gestion des déchets communaux."
    ),
    "Autre": (
        "Opérations purement techniques, fiscales ou comptables n'ayant pas de lien direct avec les piliers thématiques de la SND30. "
        "Gestion de la debt souveraine, trésorerie, nomenclature budgétaire et dotations générales de fonctionnement."
    )
}

# Keywords associated with each pillar (for hybrid or rule-based checks)
SND30_KEYWORDS = {
    "Transformation structurelle": ["industrialisation", "infrastructure", "agro-industrie", "mines", "énergie", "numérique", "port", "autoroute"],
    "Capital humain": ["éducation", "santé", "formation", "social", "emploi", "insertion", "université", "hôpital"],
    "Gouvernance": ["administration", "justice", "sécurité", "police", "défense", "corruption", "réforme", "institution"],
    "Développement régional": ["région", "commune", "local", "rural", "ctd", "décentralisation", "mairie", "aménagement"],
    "Autre": ["fiscalité", "impôt", "dette", "trésorerie", "comptabilité", "virement", "nomenclature"]
}

# ============================================================================
# NLP MODELS CONFIGURATION
# ============================================================================

MODEL_NAME = "camembert-base"
SENTENCE_BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Configuration explicite du fine-tuning
TRAINING_MODE = "fine-tune"  
USE_PRETRAINED_WEIGHTS = True
TASK_TYPE = "sequence_classification"
NUM_LABELS = len(SND30_PILLARS)

# Model parameters
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01

# Fine-tuning parameters
FINE_TUNE = True
FREEZE_LAYERS = 6
GRADIENT_ACCUMULATION_STEPS = 2

# ============================================================================
# EXTRACTION & FINANCIAL CONFIGURATION
# ============================================================================

# PDF Extraction
PDF_EXTRACTION_METHOD = "pdfplumber"

# Budget Column Identifiers (Target in LF/BIP PDFs)
COL_LIBELLE = "Libellé du Projet"
COL_MONTANT = "Montant Alloué"

# Preprocessing
REMOVE_STOPWORDS = True
LEMMATIZE = True
LOWERCASE = True
LANGUAGE = "fr"

# ============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# ============================================================================

# Similarity thresholds
SIMILARITY_THRESHOLD_HIGH = 0.85
SIMILARITY_THRESHOLD_MEDIUM = 0.70 # Utilisé pour la classification par défaut
SIMILARITY_THRESHOLD_LOW = 0.50

# Significance tests
ALPHA = 0.05
CORRELATION_METHOD = "spearman"  # Recommandé pour Corrélation Sémantique vs Financière

# ============================================================================
# YEARS AND DOCUMENTS (Updated for 2024 & 2025)
# ============================================================================

YEARS = {
    "baseline": 2024,
    "target": 2025
}

# Assurez-vous que ces fichiers existent dans data/raw/
DOCUMENT_NAMES = {
    2024: "loi_finances_2024.pdf",
    2025: "loi_finances_2025.pdf"
}

# ============================================================================
# DASHBOARD & VISUALIZATION
# ============================================================================

PILLAR_COLORS = {
    "Transformation structurelle": "#1f77b4",
    "Capital humain": "#ff7f0e",
    "Gouvernance": "#2ca02c",
    "Développement régional": "#d62728",
    "Autre": "#9467bd"
}

DASHBOARD_TITLE = "Baromètre de Glissement Sémantique - Cameroun"
DASHBOARD_SUBTITLE = "Audit IA de Conformité SND30 : Lois de Finances 2024-2025"

# ============================================================================
# HARDWARE & REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42

def detect_device():
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    except ImportError:
        return "cpu"

DEVICE = detect_device()

# ============================================================================
# UTILS & VALIDATION
# ============================================================================

def validate_methodology():
    """Valide la conformité pour le projet ISE3"""
    print("\n" + "=" * 50)
    print("VÉRIFICATION MÉTHODOLOGIQUE DU PROJET")
    print("-" * 50)
    print(f"Modèle Sémantique : {SENTENCE_BERT_MODEL}")
    print(f"Analyse Années    : {list(YEARS.values())}")
    print(f"Device de Calcul  : {DEVICE}")
    print(f"Classification    : {TRAINING_MODE}")
    print("=" * 50 + "\n")

if __name__ == "__main__":
    validate_methodology()