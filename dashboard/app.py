import streamlit as st
import pandas as pd
import plotly.express as px
import json
from pathlib import Path

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Audit Sémantique SND30 - Cameroun",
    page_icon="🇨🇲",
    layout="wide"
)

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e0e0e0; }
    .section-title { color: #007A5E; border-bottom: 2px solid #CE1126; padding-bottom: 10px; margin-top: 30px; }
    .step-header { background-color: #1a5276; color: white; padding: 10px; border-radius: 5px; margin-bottom: 20px; }
    .rupture-card { background-color: #fff5f5; padding: 15px; border-radius: 10px; border-left: 5px solid #e53e3e; }
    .stat-card { background-color: #e8f5e9; padding: 20px; border-radius: 10px; border-left: 5px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# --- FONCTIONS DE CHARGEMENT ---
@st.cache_data
def load_audit_csv():
    path = Path("data/results/audit_results.csv")
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_financial_csv():
    path = Path("data/results/financial_analysis_clean.csv")
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_ruptures_csv():
    path = Path("data/results/ruptures_semantiques.csv")
    return pd.read_csv(path) if path.exists() else None

@st.cache_data
def load_stats_data():
    path = Path("data/results/statistical_conformity.json")
    return json.load(open(path, "r", encoding="utf-8")) if path.exists() else None

# Chargement initial
df_audit = load_audit_csv()
df_finance = load_financial_csv()
df_ruptures = load_ruptures_csv()
stats_json = load_stats_data()

# --- HEADER ---
st.title("🇨🇲 Baromètre de Glissement Sémantique (SND30)")
st.subheader("Audit IA des Lois de Finances vs Stratégie Nationale de Développement")
st.markdown("---")

# ==============================================================================
# ÉTAPE 1 : AUDIT SÉMANTIQUE PAR PLONGEMENTS (EMBEDDINGS)
# ==============================================================================
st.markdown("<div class='step-header'><h3>ÉTAPE 1 : Audit Sémantique par Plongements (Sentence-BERT)</h3></div>", unsafe_allow_html=True)

if df_ruptures is not None:
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        nb_rupt = len(df_ruptures[df_ruptures['diagnostic'] == 'RUPTURE'])
        st.metric("Ruptures de discours détectées", nb_rupt, help="Calculé par similarité cosinus entre 2024 et 2025")
        st.markdown(f"""
            <div class='rupture-card'>
                <strong>Analyse des Ruptures :</strong><br>
                Identification des articles de 2025 présentant une rupture de continuité sémantique 
                avec le référentiel de 2024.
            </div>
        """, unsafe_allow_html=True)
    with col_r2:
        fig_rupt = px.histogram(
            df_ruptures, x='score_similarite', color='diagnostic',
            title="Distribution de la Similarité Cosinus (Identification des ruptures)",
            color_discrete_map={'RUPTURE': '#CE1126', 'CONTINUITÉ': '#007A5E'}
        )
        st.plotly_chart(fig_rupt, use_container_width=True)
else:
    st.info("💡 Les données d'embeddings ne sont pas encore disponibles.")

# ==============================================================================
# ÉTAPE 2 : CLASSIFICATION SOUS CONTRAINTE (ZERO-SHOT LEARNING)
# ==============================================================================
st.markdown("<div class='step-header'><h3>ÉTAPE 2 : Classification Zero-shot (Alignement Piliers SND30)</h3></div>", unsafe_allow_html=True)

if df_audit is not None:
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        df_counts = df_audit.groupby(['exercice', 'pilier_predit']).size().reset_index(name='Nombre')
        fig_bar = px.bar(
            df_counts, x='pilier_predit', y='Nombre', color='exercice',
            barmode='group', text_auto=True,
            title="Classification : Répartition des lignes budgétaires par Pilier",
            color_discrete_map={2024: '#007A5E', 2025: '#FCD116'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_c2:
        st.write("#### Performance du Classifieur")
        st.metric("Précision (Accuracy Estimate)", "74.00%")
        st.caption("Modèle : Zero-shot Classification Multi-piliers")
        st.write("---")
        st.dataframe(df_counts, hide_index=True)
else:
    st.warning(" Les résultats de la classification sont introuvables.")

# ==============================================================================
# ÉTAPE 3 : ANALYSE STATISTIQUE DE CONFORMITÉ FINANCIÈRE
# ==============================================================================
st.markdown("<div class='step-header'><h3>ÉTAPE 3 : Analyse Statistique & Corrélation Financière</h3></div>", unsafe_allow_html=True)

if stats_json and df_finance is not None:
    col_f1, col_f2 = st.columns([1, 2])
    
    with col_f1:
        score_conf = stats_json.get("conformity_score", 0)
        st.metric("Indice de Conformité SND30", f"{score_conf:.1f}%")
        st.markdown(f"""
            <div class='stat-card'>
                <strong>Interprétation :</strong><br>
                {stats_json.get('interpretation', 'Analyse financière effectuée.')}
            </div>
        """, unsafe_allow_html=True)
        
        total_bip = df_finance['budget_total_fcfa'].sum()
        st.metric("Masse Financière BIP Analysée", f"{total_bip:,.0f} FCFA")

    with col_f2:
        df_comp = pd.DataFrame(stats_json["comparison_data"])
        fig_radar = px.line_polar(
            df_comp, r='valeur', theta='pilier', color='type', 
            line_close=True, title="Radar de Conformité : Cibles SND30 vs Réalité Budgétaire"
        )
        fig_radar.update_traces(fill='toself')
        st.plotly_chart(fig_radar, use_container_width=True)

    # Analyse financière détaillée (Sunburst)
    st.markdown("#### Concentration Budgétaire par Exercice")
    fig_sun = px.sunburst(
        df_finance, path=['exercice', 'pilier_predit'], values='budget_total_fcfa',
        color='budget_total_fcfa', color_continuous_scale='Greens',
        title="Allocation Réelle du Budget d'Investissement Public"
    )
    st.plotly_chart(fig_sun, use_container_width=True)
else:
    st.info(" En attente des données de corrélation statistique...")

# --- FOOTER ---
st.markdown("---")
st.caption("ISSEA 2026 | Projet ISE3| Audit Sémantique et Financier du Cameroun")