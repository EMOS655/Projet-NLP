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

# Detailed descriptions for zero-shot classification
SND30_DESCRIPTIONS = {
    "Transformation structurelle": (
        "Croissance économique, industrialisation, développement des infrastructures routières et portuaires, "
        "production agricole et agro-industrielle, électrification, barrages hydroélectriques, mines et ressources naturelles, "
        "exportations, économie numérique, télécommunications, innovation technologique, recherche et développement, "
        "investissements productifs, zones économiques spéciales, partenariats public-privé pour l'infrastructure"
    ),
    "Capital humain": (
        "Éducation primaire et secondaire, enseignement supérieur, formation professionnelle et technique, "
        "bourses d'études, construction d'écoles et universités, santé publique, hôpitaux et centres de santé, "
        "médicaments et vaccins, protection sociale, pensions et retraites, assurance maladie, "
        "lutte contre le VIH/SIDA et le paludisme, nutrition et sécurité alimentaire, "
        "emploi des jeunes, entrepreneuriat, microfinance, programmes sociaux"
    ),
    "Gouvernance": (
        "Administration publique, modernisation de l'État, fonction publique, décentralisation et déconcentration, "
        "transfert de compétences aux collectivités territoriales, justice et tribunaux, police et gendarmerie, "
        "armée et défense nationale, sécurité intérieure, lutte contre la corruption et le détournement, "
        "transparence budgétaire, audit et contrôle, réformes institutionnelles, élections, "
        "droits de l'homme, État de droit, services publics, digitalisation administrative"
    ),
    "Développement régional": (
        "Aménagement du territoire, équilibre régional entre les régions du Cameroun, "
        "développement des zones rurales et villages, électrification rurale, routes et pistes rurales, "
        "adduction d'eau en milieu rural, développement local, projets communautaires, "
        "collectivités territoriales décentralisées, communes et communautés, "
        "réduction des disparités régionales, développement du Grand Nord, de l'Est, du Sud, "
        "infrastructures régionales, marchés locaux, agriculture vivrière"
    ),
    "Autre": (
        "Dispositions budgétaires générales, aspects techniques et financiers, procédures comptables, "
        "fiscalité et impôts, dette publique, gestion budgétaire, transferts financiers, "
        "dispositions transitoires, révisions budgétaires, crédits supplémentaires, virements de crédits, "
        "nomenclature budgétaire, opérations de trésorerie, emprunts et prêts, garanties de l'État, "
        "articles de procédure sans lien direct avec les priorités du SND30"
    )
}

# Keywords associated with each pillar
SND30_KEYWORDS = {
    "Transformation structurelle": [
        "industrialisation", "infrastructure", "énergie", "agriculture",
        "numérique", "innovation", "technologie", "transport", "manufacture",
        "production", "économie", "secteur primaire", "secteur secondaire"
    ],
    "Capital humain": [
        "éducation", "santé", "formation", "emploi", "école", "université",
        "hôpital", "social", "compétence", "qualification", "jeunesse",
        "enseignement", "médical", "ressources humaines"
    ],
    "Gouvernance": [
        "administration", "gouvernement", "justice", "sécurité", "institution",
        "réforme", "transparence", "corruption", "décentralisation", "police",
        "armée", "défense", "magistrature", "état de droit"
    ],
    "Développement régional": [
        "région", "territoire", "local", "communal", "rural", "urbain",
        "aménagement", "collectivité", "municipalité", "décentralisé",
        "développement local", "équilibre territorial"
    ],
    "Autre": [
        "budget", "crédit", "impôt", "fiscalité", "dette", "emprunt",
        "trésorerie", "comptabilité", "financier", "procédure", "technique",
        "disposition", "transfert", "nomenclature", "révision"
    ]
}


# ============================================================================
# NLP MODELS CONFIGURATION
# ============================================================================

# Model selection
MODEL_NAME = "camembert-base"
SENTENCE_BERT_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Configuration explicite du fine-tuning
TRAINING_MODE = "fine-tune"  # Options: "fine-tune", "pretrained-only", "from-scratch"
USE_PRETRAINED_WEIGHTS = True  # Utiliser les poids pré-entraînés de CamemBERT
TASK_TYPE = "sequence_classification"  # Type de tâche pour le fine-tuning

# Nombre de labels (5 piliers SND30 incluant "Autre")
NUM_LABELS = len(SND30_PILLARS)  # 5 piliers (4 SND30 + Autre)

# Alternative models
ALTERNATIVE_MODELS = {
    "camembert_base": "camembert-base",
    "camembert_large": "camembert-large",
    "flaubert_base": "flaubert/flaubert_base_cased",
    "bert_multilingual": "bert-base-multilingual-cased",
    "sentence_bert_fr": "dangvantuan/sentence-camembert-base"
}

# Model parameters
MAX_LENGTH = 512
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5  # Augmenté pour un meilleur fine-tuning
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01  # Régularisation

# Fine-tuning parameters
FINE_TUNE = True
FREEZE_LAYERS = 6  # Geler les 6 premières couches de CamemBERT
GRADIENT_ACCUMULATION_STEPS = 2  # Pour les petits GPU


# ============================================================================
# TRAINING DATA CONFIGURATION
# ============================================================================

# Stratégie de création du dataset d'entraînement
TRAINING_DATA_STRATEGY = "labeled_examples"  # Options: "labeled_examples", "weak_supervision", "zero-shot-bootstrap"

# Split des données
TRAIN_TEST_SPLIT = 0.8
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

# Augmentation de données
USE_DATA_AUGMENTATION = True
AUGMENTATION_TECHNIQUES = ["synonym_replacement", "back_translation", "paraphrase"]

# Labeled data path (créer des exemples annotés manuellement)
LABELED_DATA_PATH = LABELED_DATA_DIR / "snd30_labeled_examples.csv"

# Minimum d'exemples par pilier pour le fine-tuning
MIN_EXAMPLES_PER_CLASS = 50  # Minimum recommandé: 50-100 par classe

# Stratified sampling
USE_STRATIFIED_SPLIT = True


# ============================================================================
# MODEL CHECKPOINTING AND SAVING
# ============================================================================

# Sauvegarde des modèles fine-tunés
SAVE_MODEL = True

# Stratégie de sauvegarde
SAVE_STRATEGY = "epoch"  # Options: "epoch", "steps", "best"
SAVE_TOTAL_LIMIT = 3  # Garder seulement les 3 meilleurs checkpoints
LOAD_BEST_MODEL_AT_END = True

# Évaluation pendant l'entraînement
EVALUATION_STRATEGY = "epoch"  # Évaluer à chaque epoch
METRIC_FOR_BEST_MODEL = "f1"  # Utiliser F1-score pour sélectionner le meilleur modèle

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 3
EARLY_STOPPING_THRESHOLD = 0.001


# ============================================================================
# PREPROCESSING CONFIGURATION
# ============================================================================

# PDF extraction
PDF_EXTRACTION_METHOD = "pdfplumber"

# Text cleaning
REMOVE_STOPWORDS = True
LEMMATIZE = True
LOWERCASE = True
REMOVE_NUMBERS = False
REMOVE_PUNCTUATION = False

# Language
LANGUAGE = "fr"
SPACY_MODEL = "fr_core_news_md"

# Segmentation
MIN_SENTENCE_LENGTH = 10
MAX_SENTENCE_LENGTH = 100
CHUNK_SIZE = 1000


# ============================================================================
# SIMILARITY AND CLUSTERING CONFIGURATION
# ============================================================================

# Similarity thresholds
SIMILARITY_THRESHOLD_HIGH = 0.85
SIMILARITY_THRESHOLD_MEDIUM = 0.70
SIMILARITY_THRESHOLD_LOW = 0.50

# Semantic drift detection
DRIFT_THRESHOLD = 0.30

# Clustering
CLUSTERING_METHOD = "HDBSCAN"
N_CLUSTERS = 10
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3

# Dimensionality reduction
USE_DIMENSIONALITY_REDUCTION = True
REDUCTION_METHOD = "UMAP"
N_COMPONENTS = 2


# ============================================================================
# CLASSIFICATION CONFIGURATION
# ============================================================================

# Choix de la méthode de classification
CLASSIFICATION_METHOD = "fine-tuned-camembert"  # Options: "fine-tuned-camembert", "zero-shot", "hybrid"

# Zero-shot classification (comme fallback)
# Modèles LÉGERS et RAPIDES pour le français
ZERO_SHOT_MODEL = "joeddav/xlm-roberta-base-xnli"  # LÉGER (~560 MB) - Multilingue excellent pour français
ZERO_SHOT_MULTILINGUAL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  # Alternative si besoin

# Autres options de modèles zero-shot par TAILLE
ALTERNATIVE_ZERO_SHOT_MODELS = {
    # LÉGERS (< 600 MB) - RECOMMANDÉS
    "xlm_roberta_base": "joeddav/xlm-roberta-base-xnli",  # 560 MB - MEILLEUR CHOIX
    "mbart_small": "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli",  # ~760 MB
    
    # MOYENS (600 MB - 1.5 GB)
    "deberta_multilingual": "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",  # ~760 MB
    
    # LOURDS (> 2 GB) - À éviter si mémoire limitée
    "xlm_roberta_large": "joeddav/xlm-roberta-large-xnli",  # 2.24 GB - LOURD !
    "bart_mnli": "facebook/bart-large-mnli",  # 1.63 GB - Anglais uniquement
}

USE_MULTILINGUAL = True  # Activé car on utilise des modèles multilingues

# Classification threshold
CLASSIFICATION_CONFIDENCE_THRESHOLD = 0.5

# Ensemble method (optionnel)
USE_ENSEMBLE = False  # Combiner fine-tuned + zero-shot
ENSEMBLE_WEIGHTS = {"fine_tuned": 0.7, "zero_shot": 0.3}


# ============================================================================
# STATISTICAL ANALYSIS CONFIGURATION
# ============================================================================

# Significance tests
ALPHA = 0.05
CORRELATION_METHOD = "pearson"

# Bootstrap parameters
N_BOOTSTRAP = 1000
BOOTSTRAP_CONFIDENCE = 0.95


# ============================================================================
# YEARS AND DOCUMENTS
# ============================================================================

YEARS = {
    "baseline": 2024,
    "target": 2025
}

DOCUMENT_NAMES = {
    2024: "Loi_de_Finances_2024.pdf",
    2025: "Loi_de_Finances_2025.pdf",
    "SND30": "SND30_Document.pdf"
}


# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Color scheme for pillars
PILLAR_COLORS = {
    "Transformation structurelle": "#1f77b4",
    "Capital humain": "#ff7f0e",
    "Gouvernance": "#2ca02c",
    "Développement régional": "#d62728",
    "Autre": "#9467bd"  # Violet pour "Autre"
}

# Plot settings
FIGURE_SIZE = (12, 6)
DPI = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid"


# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================

# Streamlit settings
DASHBOARD_TITLE = "Baromètre de Glissement Sémantique - Lois de Finances Cameroun"
DASHBOARD_SUBTITLE = "Analyse NLP des Lois de Finances 2024-2026 vs SND30"
DASHBOARD_ICON = "📊"

# Update frequency
CACHE_TTL = 3600


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_LEVEL = "INFO"
LOG_FILE = LOGS_DIR / "nlp_project.log"


# ============================================================================
# EVALUATION METRICS
# ============================================================================

# Metrics to compute
METRICS = [
    "f1_score",
    "precision",
    "recall",
    "accuracy",
    "log_loss"
]

# Cross-validation
CV_FOLDS = 5


# ============================================================================
# RANDOM SEED (for reproducibility)
# ============================================================================

RANDOM_SEED = 42


# ============================================================================
# HARDWARE CONFIGURATION (détection automatique)
# ============================================================================

# Device configuration - sera déterminé au moment de l'import de torch
DEVICE = "cpu"  # Par défaut CPU, sera mis à jour quand torch est importé
NUM_WORKERS = 4
USE_FP16 = False

def detect_device():
    """Détecte automatiquement le device disponible"""
    global DEVICE, USE_FP16
    try:
        import torch
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        USE_FP16 = torch.cuda.is_available()
        print(f"Device détecté: {DEVICE}")
        if DEVICE == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch pas encore installé. Device: CPU par défaut")
        DEVICE = "cpu"
        USE_FP16 = False
    return DEVICE


# ============================================================================
# API KEYS AND AUTHENTICATION
# ============================================================================

# Hugging Face Token
# Nécessaire uniquement pour:
# - Télécharger des modèles privés ou gated
# - Augmenter les limites de téléchargement
# Pour obtenir un token: https://huggingface.co/settings/tokens
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", None)

# OpenAI API Key
# Nécessaire uniquement si vous utilisez GPT ou d'autres modèles OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

# NOTE: Pour CamemBERT-base, aucun token n'est nécessaire (modèle public)
# Si vous obtenez des erreurs 401/403, vous pouvez définir HF_TOKEN comme:
# HF_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"


# ============================================================================
# METHODOLOGY VALIDATION
# ============================================================================

def validate_methodology():
    """Valide que la configuration respecte la méthodologie CamemBERT fine-tuning"""
    issues = []
    warnings = []
    
    # Vérifications critiques
    if not USE_PRETRAINED_WEIGHTS:
        issues.append("❌ Les poids pré-entraînés de CamemBERT ne sont pas activés")
    
    if not FINE_TUNE:
        issues.append("❌ Le fine-tuning n'est pas activé")
    
    if CLASSIFICATION_METHOD == "zero-shot":
        issues.append("❌ Méthode zero-shot utilisée au lieu du fine-tuning")
    
    if TRAINING_MODE != "fine-tune":
        issues.append(f"❌ TRAINING_MODE est '{TRAINING_MODE}' au lieu de 'fine-tune'")
    
    # Vérifications d'avertissement
    if NUM_EPOCHS < 3:
        warnings.append(f"⚠️  Nombre d'epochs faible ({NUM_EPOCHS}). Recommandé: 5-10")
    
    if not LABELED_DATA_PATH.exists():
        warnings.append(f"⚠️  Pas de données labellisées trouvées à {LABELED_DATA_PATH}")
        warnings.append("    → Vous devrez créer ce fichier pour le fine-tuning")
    
    if BATCH_SIZE < 8:
        warnings.append(f"⚠️  Batch size très petit ({BATCH_SIZE}). Recommandé: 16-32")
    
    if not SAVE_MODEL:
        warnings.append("⚠️  La sauvegarde du modèle n'est pas activée")
    
    # Affichage des résultats
    print("\n" + "=" * 80)
    print("VALIDATION DE LA MÉTHODOLOGIE CAMEMBERT FINE-TUNING")
    print("=" * 80)
    
    if issues:
        print("\n🚨 PROBLÈMES CRITIQUES DÉTECTÉS:")
        for issue in issues:
            print(f"  {issue}")
    
    if warnings:
        print("\n⚠️  AVERTISSEMENTS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ Configuration CONFORME à la méthodologie CamemBERT fine-tuning")
        print("\nParamètres clés:")
        print(f"  • Modèle: {MODEL_NAME}")
        print(f"  • Mode d'entraînement: {TRAINING_MODE}")
        print(f"  • Méthode de classification: {CLASSIFICATION_METHOD}")
        print(f"  • Nombre de labels: {NUM_LABELS}")
        print(f"  • Epochs: {NUM_EPOCHS}")
        print(f"  • Batch size: {BATCH_SIZE}")
        print(f"  • Learning rate: {LEARNING_RATE}")
    elif not issues:
        print("\n✅ Configuration de base CORRECTE")
        print("   Quelques améliorations recommandées ci-dessus")
    else:
        print("\n❌ Configuration NON CONFORME")
        print("\nActions requises:")
        print("  1. Corriger les problèmes critiques ci-dessus")
        print("  2. Créer un fichier de données labellisées pour l'entraînement")
        print("  3. Relancer la validation")
    
    print("=" * 80 + "\n")
    
    return len(issues) == 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_config_summary():
    """Print a summary of the current configuration"""
    print("=" * 80)
    print("NLP PROJECT CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Training Mode: {TRAINING_MODE}")
    print(f"Classification Method: {CLASSIFICATION_METHOD}")
    print(f"Device: {DEVICE}")
    print(f"SND30 Pillars: {len(SND30_PILLARS)}")
    print(f"Years analyzed: {list(YEARS.values())}")
    print(f"Similarity Threshold: {SIMILARITY_THRESHOLD_MEDIUM}")
    print(f"Clustering Method: {CLUSTERING_METHOD}")
    print(f"Fine-tuning: {'Enabled' if FINE_TUNE else 'Disabled'}")
    print(f"Number of Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("=" * 80)


def create_labeled_data_template():
    """Crée un template CSV pour les données labellisées"""
    import pandas as pd
    
    # Template avec quelques exemples
    template_data = {
        'text': [
            "Construction de routes et d'infrastructures de transport",
            "Formation professionnelle des jeunes",
            "Renforcement de l'administration publique",
            "Développement des zones rurales"
        ],
        'label': [
            "Transformation structurelle",
            "Capital humain",
            "Gouvernance",
            "Développement régional"
        ],
        'confidence': [1.0, 1.0, 1.0, 1.0]
    }
    
    df = pd.DataFrame(template_data)
    
    # Créer le fichier si il n'existe pas
    if not LABELED_DATA_PATH.exists():
        df.to_csv(LABELED_DATA_PATH, index=False, encoding='utf-8')
        print(f"\n✅ Template de données labellisées créé: {LABELED_DATA_PATH}")
        print(f"   Ajoutez au moins {MIN_EXAMPLES_PER_CLASS} exemples par pilier\n")
        return True
    else:
        print(f"\n⚠️  Le fichier existe déjà: {LABELED_DATA_PATH}\n")
        return False


if __name__ == "__main__":
    # Détecte le device si possible
    detect_device()
    
    # Affiche le résumé
    get_config_summary()
    
    # Valide la méthodologie
    validate_methodology()
    
    # Propose de créer le template
    try:
        create_labeled_data_template()
    except Exception as e:
        print(f"Note: Installez pandas pour créer le template automatiquement")
        print(f"      pip install pandas")