# IA Audit de Glissement Sémantique : LF 2024-2025 vs SND30
**Intelligence Artificielle et Finances Publiques — Projet NLP ISE3 | ISSEA Yaoundé**

> Utilisation des modèles Transformers pour l'audit sémantique et la classification budgétaire des Lois de Finances du Cameroun au regard de la Stratégie Nationale de Développement 2020-2030 (SND30).

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

##  Table des Matières

1. [Description du Projet](#-description-du-projet)
2. [Problématique](#-problématique)
3. [Pipeline Analytique](#-pipeline-analytique-méthodologie)
4. [Structure du Projet](#-structure-du-projet)
5. [Prérequis](#-prérequis)
6. [Installation](#-installation)
7. [Données Requises](#-données-requises)
8. [Guide d'Exécution](#️-guide-dexécution)
9. [Architecture des Résultats](#-architecture-des-résultats)
10. [Technologies Utilisées](#-technologies-utilisées)
11. [Métriques de Performance](#-métriques-de-performance)
12. [Tests](#-tests)
13. [Auteurs](#-auteurs)

---

##  Description du Projet

Ce projet vise à quantifier le glissement sémantique entre les intentions stratégiques de la SND30 et les allocations budgétaires réelles des Lois de Finances du Cameroun (2024-2025). Il mobilise des techniques avancées de Traitement du Langage Naturel (NLP) et des modèles Transformers pour mesurer l'évolution des priorités budgétaires de l'État et vérifier leur alignement avec la Stratégie Nationale de Développement 2020-2030 (SND30).

### Contexte

Dans le cadre de la SND30, la Loi de Finances constitue l'instrument pivot de la politique économique du Cameroun. La complexité et le volume des documents textuels rendent difficile l'évaluation rapide de la cohérence sémantique entre les budgets annuels et les objectifs stratégiques de l'État.

---

##  Problématique

**Comment l'intelligence artificielle, à travers le NLP, peut-elle mesurer mathématiquement l'évolution des priorités de l'État camerounais entre la Loi de Finances 2024 et 2025 ?**

Existe-t-il un alignement statistiquement significatif entre le discours budgétaire annuel et les piliers de la SND30 ?

---

##  Pipeline Analytique (Méthodologie)

Le projet est structuré en **trois piliers scientifiques majeurs**, précédés d'une phase intensive de traitement des données.

###  Phase 0 — Traitement Intensif des Données

Avant toute analyse, les données subissent une préparation rigoureuse pour garantir l'intégrité des résultats :

- **Extraction de Structure** (`extractor.py`) : Récupération des données tabulaires et textuelles à partir des documents PDF officiels.

### 1.  Audit Sémantique par Plongements (Embeddings)

- **Technologie** : Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
- **Logique** : Transformation des articles de loi en vecteurs numériques de haute dimension.
- **Analyse** : Calcul de la similarité cosinus entre les articles de 2024 et 2025 pour détecter les ruptures de discours (scores inférieurs à **0.70**).

### 2.  Classification sous Contrainte (Zero-Shot Learning)

- **Technologie** : Inférence sémantique (NLI)
- **Logique** : Classification automatique des lignes de dépenses dans les **4 piliers SND30** :
  - Transformation structurelle de l'économie
  - Développement du capital humain
  - Gouvernance
  - Développement régional
  -Autre

### 3.  Analyse Statistique de Conformité

- **Technologie** : Corrélation financière et tests du Khi-deux
- **Logique** : Croisement des fréquences thématiques prédites par l'IA avec les montants financiers (FCFA) du Budget d'Investissement Public (BIP). Calcul d'un score de conformité global.

---

##  Structure du Projet

```
Projet-NLP/
│
├── README.md                          # Ce fichier
├── LICENSE                            # Licence du projet
├── .gitignore                         # Fichiers à ignorer par Git
├── requirements.txt                   # Dépendances Python
├── config.py                          # Configuration centrale du projet
├── env_config.py                      # Gestion des variables d'environnement
├── Diagnostic_donnees.py              # Script de diagnostic des données
│
├── data/                              # Données du projet
│   ├── raw/                           # PDFs bruts (documents officiels)
│   │   ├── Loi_de_Finances_2024.pdf
│   │   ├── Loi_de_Finances_2025.pdf
│   │   └── SND30_Document.pdf
│   ├── labeled/                       # Données annotées
│   ├── processed/                     # Données nettoyées et prétraitées
│   ├── test/                          # Données de test
│   └── results/                       # Sorties des analyses IA
│       ├── audit_results.csv          # Résultats de classification (piliers)
│       ├── ruptures_semantiques.csv   # Détection des ruptures de discours
│       ├── financial_analysis_clean.csv # Corrélation budgétaire nettoyée
│       ├── statistical_conformity.json  # Scores et radar de conformité SND30
│       ├── all_processed_data.json    # Données consolidées + métriques
│       └── roc_curves.json            # Courbes ROC par pilier
│
├── src/                               # Code source principal
│   ├── __init__.py                    # Initialisation du package
│   ├── extrator.py                    # Extraction des textes et tableaux PDF
│   ├── main.py                        # Nettoyage, normalisation, lemmatisation
│   ├── embeddings_engine.py           # Similarité cosinus & détection de ruptures
│   ├── classifier.py                  # Labels et logique de classification
│   ├── semantic_engine.py             # Classification multi-piliers SND30
│   ├── financial_correlation.py       # Corrélation prédictions IA / montants FCFA
│   ├── stat_analysis.py               # Scores de conformité, Chi-deux, radar
│   └── utils.py                       # Fonctions utilitaires partagées
│
├── dashboard/                         # Application Streamlit
│   └── app.py                         # Dashboard interactif (3 onglets)
│
├── notebooks/                         # Jupyter Notebooks exploratoires
│   ├── 01_exploration.ipynb           # Exploration des données brutes
│   ├── 02_preprocessing.ipynb         # Prétraitement détaillé
│   └── 03_modelisation.ipynb          # Modélisation et analyse
│
├── models/                            # Modèles sauvegardés
│   ├── pretrained/                    # Modèles pré-entraînés
│   └── finetuned/                     # Modèles fine-tunés (si applicable)
│
├── outputs/                           # Outputs générés
│   ├── figures/                       # Graphiques et visualisations
│   ├── tables/                        # Tableaux de résultats
│   └── reports/                       # Rapports générés
│
├── tests/                             # Tests unitaires
├── logs/                              # Fichiers de logs
│   └── nlp_project.log
└── nlpenv/                            # Environnement virtuel Python (ignoré par Git)
```

---

##  Prérequis

### Logiciels
- **Python** : Version 3.8 ou supérieure
- **Git** : Pour cloner le repository
- **GPU (optionnel)** : NVIDIA GPU avec CUDA pour accélération (recommandé)

### Connaissances
- Bases en Python
- Notions de NLP et Machine Learning
- Compréhension des modèles Transformers (recommandé)

---

##  Installation

### 1. Cloner le Repository

```bash
git clone : https://github.com/EMOS655/Projet-NLP.git
cd Projet-NLP
```

### 2. Créer l'Environnement Virtuel

```bash
python -m venv nlpenv

# Windows PowerShell :
.\nlpenv\Scripts\Activate.ps1

# Windows CMD :
nlpenv\Scripts\activate.bat

# Linux/Mac :
source nlpenv/bin/activate
```

> **Note Windows** : En cas d'erreur de politique d'exécution, exécutez :
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. Installer les Dépendances

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Télécharger le Modèle SpaCy Français

```bash
python -m spacy download fr_core_news_md
```

### 5. Vérifier l'Installation

```bash
python config.py
```

---

##  Données Requises

Placez les documents suivants dans `data/raw/` avant toute exécution :

| Fichier | Description | Source |
|---|---|---|
| `Loi_de_Finances_2024.pdf` | Loi de Finances 2024 du Cameroun | [MINFI](https://www.minfi.cm) |
| `Loi_de_Finances_2025.pdf` | Loi de Finances 2025 du Cameroun | [MINFI](https://www.minfi.cm) |
| `SND30_Document.pdf` | Stratégie Nationale de Développement 2020-2030 | [MINEPAT](https://www.minepat.gov.cm) |

---

##  Guide d'Exécution

Les scripts doivent être exécutés **dans l'ordre suivant** pour garantir la génération correcte des données du Dashboard.

### Étape A — Extraction & Nettoyage

```bash
# 1. Extraction brute des textes et tableaux depuis les PDFs
python src/extrator.py


```

### Étape B — Moteurs d'Analyse IA

```bash
# 3. Génère l'analyse des ruptures sémantiques → ruptures_semantiques.csv
python src/embeddings_engine.py

# 4. Initialise les labels et la logique de classification
python src/classifier.py

# 5. Exécute la classification multi-piliers SND30 → audit_results.csv
python src/semantic_engine.py
```
### Execution de toutes les etapes
python src/main.py

### Étape C — Corrélation & Statistiques

```bash
# 6. Lie les prédictions IA aux montants financiers → financial_analysis_clean.csv
python src/financial_correlation.py

# 7. Calcule les scores finaux et le radar de conformité → statistical_conformity.json
python src/stat_analysis.py
```

### Étape D — Visualisation

```bash
# 8. Lance le Dashboard interactif
streamlit run dashboard/app.py
```

Le dashboard s'ouvrira à : `http://localhost:8501`

---

##  Architecture des Résultats

Le Dashboard Streamlit (3 onglets) consomme les sorties suivantes dans `data/results/` :

| Fichier | Onglet Dashboard | Contenu |
|---|---|---|
| `ruptures_semantiques.csv` |  Audit Sémantique | Score de similarité cosinus et diagnostic RUPTURE / CONTINUITÉ |
| `audit_results.csv` |  Classification | Répartition volumétrique et thématique des projets par pilier |
| `financial_analysis_clean.csv` |  Analyse Financière | Montants FCFA croisés avec les piliers prédits |
| `statistical_conformity.json` |  Analyse Financière | Score de conformité SND30, test Khi-deux, données du radar |
| `all_processed_data.json` |  Classification | Métriques du modèle (Accuracy, F1, Log-Loss) |
| `roc_curves.json` |  Classification | Courbes ROC et AUC par pilier SND30 |

---

##  Technologies Utilisées

### NLP & Deep Learning
- **Transformers** (Hugging Face) — CamemBERT, Sentence-BERT (`paraphrase-multilingual-MiniLM-L12-v2`)
- **PyTorch** — Framework de deep learning
- **SpaCy** — Traitement du langage naturel pour le français

### Data Science
- **Pandas** / **NumPy** — Manipulation et calculs
- **Scikit-learn** — Machine learning et statistiques
- **SciPy** — Tests statistiques avancés (Khi-deux)

### Visualisation & Dashboard
- **Plotly** — Graphiques interactifs (histogramme, barres, ROC, radar, sunburst)
- **Streamlit** — Interface web interactive (3 onglets)

### Traitement PDF
- **pdfplumber** — Extraction de texte et tableaux depuis PDFs
- **PyPDF2** — Manipulation de fichiers PDF

---

##  Métriques de Performance

| Métrique | Description |
|---|---|
| **Accuracy** | Taux de réussite global de la classification |
| **F1-Score (Micro)** | Mesure harmonique précision/rappel |
| **Log-Loss** | Incertitude du modèle (plus bas = meilleur) |
| **AUC-ROC** | Pouvoir discriminant par pilier SND30 |
| **Similarité Cosinus** | Score de cohérence sémantique (0–1) |



##  Tests

```bash
# Tous les tests
pytest tests/

# Avec couverture
pytest --cov=src tests/
```

---

## 👥 Auteurs

**Groupe ISE3 — Promotion 2025-2026**

- **FONKOUA NGANKE Voltaire** — [Email:voltairefonkoua@gmail.com]
- **KALEFACK NCUEPI Sergeo** — [Email:kalefacksergeo@gmail.com]
- **MABIALA Michée 3** — [Email:micheemabiala99@gmail.com]
- **SOME Pascal** — [Email:student.pascal.some@issea-cemac.org]

**Superviseur** : **MBIA NDI Marie Thérèse** — [mbialaura12@gmail.com]

---


---

##  Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.



##  Bibliographie

1. Devlin, J. et al. (2018). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. arXiv:1810.04805
2. Martin, L. et al. (2020). *CamemBERT: a Tasty French Language Model*. Proceedings of ACL 2020.
3. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP-IJCNLP 2019.
4. République du Cameroun (2020). *Document de Stratégie Nationale de Développement 2020-2030 (SND30)*.

---

##  État du Projet

- [x] Structure du projet
- [x] Configuration de base
- [x] Extraction et nettoyage des données (`extrator.py`, `main.py`)
- [x] Moteur d'embeddings et détection de ruptures (`embeddings_engine.py`)
- [x] Classification zero-shot multi-piliers (`classifier.py`, `semantic_engine.py`)
- [x] Corrélation financière (`financial_correlation.py`)
- [x] Analyses statistiques et score de conformité (`stat_analysis.py`)
- [x] Dashboard Streamlit interactif (`dashboard/app.py`)
- [x] Tests unitaires complets
- [x] Rapport technique final

---

*Développé par les étudiants ISE3 de l'ISSEA — Version 2.0.0 — Février 2026*