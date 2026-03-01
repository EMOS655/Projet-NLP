import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Audit Sémantique SND30 - Cameroun",
    page_icon="IA",
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
    /* Centrage vertical des images du header */
    [data-testid="stHorizontalBlock"] {
        align-items: center;
    }
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

@st.cache_data
def load_processed_json():
    path = Path("data/results/all_processed_data.json")
    return json.load(open(path, "r", encoding="utf-8")) if path.exists() else None

@st.cache_data
def load_roc_data():
    path = Path("data/results/roc_curves.json")
    return json.load(open(path, "r", encoding="utf-8")) if path.exists() else None

# Chargement initial
df_audit = load_audit_csv()
df_finance = load_financial_csv()
df_ruptures = load_ruptures_csv()
stats_json = load_stats_data()
processed_json = load_processed_json()
roc_json = load_roc_data()


# --- HEADER AVEC DRAPEAU ET LOGO ---
import base64

# --- FONCTION POUR ENCODER L'IMAGE LOCALE ---
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- HEADER AVEC IMAGES ROBUSTES ---
col_logo_1, col_titre, col_logo_2 = st.columns([1, 5, 1])

with col_logo_1:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Flag_of_Cameroon.svg/640px-Flag_of_Cameroon.svg.png", width=120)

with col_titre:
    st.markdown("""
        <div style='text-align: center;'>
            <h1 style='margin-bottom: 0;'>NLP Baromètre de glissement sémantique (SND30)</h1>
            <h3 style='margin-top: 0; color: #555;'>Audit IA des lois de finances vs stratégie nationale de développement</h3>
        </div>
    """, unsafe_allow_html=True)

with col_logo_2:
    try:
        st.image("https://ensai.fr/wp-content/uploads/2019/07/logo_ISSEA.png", width=250)
    except:
        st.write("Logo ISSEA")

st.markdown("---")

# ==============================================================================
# ONGLETS PRINCIPAUX
# ==============================================================================
tab_audit, tab_classification, tab_analyse = st.tabs([
    " Audit Sémantique",
    " Classification",
    " Analyse Financière"
])

# ==============================================================================
# ONGLET 1 : AUDIT SÉMANTIQUE PAR PLONGEMENTS (EMBEDDINGS)
# ==============================================================================
with tab_audit:
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
                title="Distribution de la similarité cosinus (Identification des ruptures)",
                color_discrete_map={'RUPTURE': '#CE1126', 'CONTINUITÉ': '#007A5E'}
            )
            st.plotly_chart(fig_rupt, use_container_width=True)
    else:
        st.info(" Les données d'embeddings ne sont pas encore disponibles.")

# ==============================================================================
# ONGLET 2 : CLASSIFICATION SOUS CONTRAINTE (ZERO-SHOT LEARNING) & PERFORMANCE
# ==============================================================================
with tab_classification:
    st.markdown("<div class='step-header'><h3>ÉTAPE 2 : Classification et performance du modèle</h3></div>", unsafe_allow_html=True)

    if df_audit is not None:
        if processed_json and "metrics" in processed_json:
            m = processed_json["metrics"]
            perf_1, perf_2, perf_3 = st.columns(3)
            perf_1.metric("Précision (Accuracy)", f"{m.get('accuracy', 0)*100:.2f}%")
            perf_2.metric("F1-Score (Micro)", f"{m.get('f1_micro', 0):.4f}")
            perf_3.metric("Log-Loss (Incertitude)", f"{m.get('log_loss', 0):.4f}", delta_color="inverse")

        st.markdown("---")

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            df_counts = df_audit.groupby(['exercice', 'pilier_predit']).size().reset_index(name='Nombre')
            fig_bar = px.bar(
                df_counts, x='pilier_predit', y='Nombre', color='exercice',
                barmode='group', text_auto=True,
                title="Répartition des articles par pilier selon les deux excercices",
                color_discrete_map={2024: '#007A5E', 2025: '#FCD116'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_c2:
            if roc_json:
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='grey')))
                for label, m in roc_json.items():
                    fig_roc.add_trace(go.Scatter(x=m['fpr'], y=m['tpr'], mode='lines', name=f"{label} (AUC={m['auc']})"))

                fig_roc.update_layout(
                    title="Courbes ROC",
                    xaxis_title="Taux de faux positifs", yaxis_title="Taux de vrais positifs",
                    legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
                    height=450, margin=dict(l=20, r=20, t=40, b=20)
                )
                st.plotly_chart(fig_roc, use_container_width=True)
    else:
        st.warning(" Les résultats de la classification sont introuvables.")

# ==============================================================================
# ONGLET 3 : ANALYSE STATISTIQUE DE CONFORMITÉ FINANCIÈRE
# ==============================================================================
with tab_analyse:
    st.markdown("<div class='step-header'><h3>ÉTAPE 3 : Analyse Statistique et Corrélation Financière</h3></div>", unsafe_allow_html=True)

    if stats_json and df_finance is not None:
        p_value = stats_json.get("p_value", 1.0)

        st.markdown("#### Rigueur Statistique de l'Audit")
        if p_value < 0.05:
            st.success(f" **Test du Khi-deux significatif** (p-value = {p_value:.4e})")
            st.caption("L'IA confirme que les changements de priorité budgétaire entre 2024 et 2025 sont statistiquement réels.")
        else:
            st.warning(f" **Test du Khi-deux non significatif** (p-value = {p_value:.4f})")
            st.caption("La structure budgétaire ne présente pas de changement statistiquement majeur.")

        st.markdown("---")

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
            st.metric("Masse financière du BIP analysée", f"{total_bip:,.0f} FCFA")

        with col_f2:
            df_comp = pd.DataFrame(stats_json["comparison_data"])
            fig_radar = px.line_polar(
                df_comp, r='valeur', theta='pilier', color='type',
                line_close=True, title="Radar de Conformité : Cibles SND30 vs Réalité Budgétaire"
            )
            fig_radar.update_traces(fill='toself')
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("#### Concentration Budgétaire par Exercice")
        fig_sun = px.sunburst(
            df_finance, path=['exercice', 'pilier_predit'], values='budget_total_fcfa',
            color='budget_total_fcfa', color_continuous_scale='Greens',
            title="Allocation réelle du Budget d'Investissement Public"
        )
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.info(" En attente des données de corrélation statistique...")

# --- FOOTER ---
st.markdown("---")
st.caption("ISSEA 2026 | Projet ISE3 | Audit Sémantique et Financier du Cameroun")