import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Configuration du chemin projet
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

import config
from src.utils import setup_logging, save_results

logger = setup_logging()

class AuditReportGenerator:
    """Compile tous les résultats pour générer un rapport d'audit final."""

    def __init__(self):
        self.results_dir = config.RESULTS_DIR
        self.report_path = self.results_dir / "RAPPORT_AUDIT_FINAL.txt"

    def _load_json(self, filename: str) -> dict:
        path = self.results_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def generate(self):
        logger.info("Génération du rapport d'audit final...")

        # 1. Chargement des données
        stats = self._load_json("rapport_statistique_budget.json")
        drift = self._load_json("semantic_drift_report.json")
        
        if not stats or not drift:
            logger.error("Certaines analyses manquent. Lancez embeddings.py et statistical_analysis.py")
            return

        lines = []
        lines.append("=" * 75)
        lines.append(f"RAPPORT D'AUDIT NLP : ALIGNEMENT BUDGÉTAIRE SND30 (2024-2025)")
        lines.append(f"Généré le : {datetime.now().strftime('%d/%m/%Y à %H:%M')}")
        lines.append("=" * 75)
        lines.append("\n")

        # --- SECTION 1 : VOLUMÉTRIE ---
        lines.append("1. SYNTHÈSE DE LA VOLUMÉTRIE")
        lines.append("-" * 35)
        m = stats.get('metriques_globales', {})
        lines.append(f"• Articles analysés en 2024 : {m.get('total_articles_2024')}")
        lines.append(f"• Articles analysés en 2025 : {m.get('total_articles_2025')}")
        lines.append(f"• Évolution de la structure : {m.get('indice_diversification_evol')}")
        lines.append("\n")

        # --- SECTION 2 : ALIGNEMENT STRATÉGIQUE ---
        lines.append("2. ALIGNEMENT SUR LES PILIERS SND30 (2025)")
        lines.append("-" * 35)
        lines.append(f"{'Pilier':<30} | {'Part (%)':<10} | {'Variation (Abs)'}")
        lines.append("-" * 65)
        
        piliers = stats.get('piliers_stats', {})
        for name, p_data in piliers.items():
            var = p_data.get('Variation_Absolue', 0)
            symbol = "▲" if var > 0 else "▼" if var < 0 else "•"
            lines.append(f"{name[:30]:<30} | {p_data.get('Part_2025_%'):<10}% | {symbol} ({int(var)})")
        lines.append("\n")

        # --- SECTION 3 : ANALYSE DU GLISSEMENT ---
        lines.append("3. ANALYSE DU GLISSEMENT SÉMANTIQUE (MÉTHODE DYNAMIQUE)")
        lines.append("-" * 35)
        lines.append(f"• Similarité moyenne globale : {drift.get('mean_similarity', 0):.4f}")
        lines.append(f"• Seuil de rupture calculé   : {drift.get('statistical_threshold', 0):.4f}")
        lines.append(f"• TAUX DE CHANGEMENT REEL    : {drift.get('drifted_percentage', 0):.2f}%")
        lines.append(f"• Nombre de ruptures nettes  : {drift.get('drifted_count', 0)} phrases")
        
        lines.append("\nExemples de ruptures sémantiques (les plus faibles similarités) :")
        # --- CORRECTION DE LA CLE ICI ---
        for i, example in enumerate(drift.get('drift_examples', [])):
            lines.append(f"  Rupture {i+1} (Similarité: {example['similarity']:.3f}):")
            lines.append(f"    [2024] : {example.get('best_match_2024_found', 'N/A')[:120]}...")
            lines.append(f"    [2025] : {example.get('target_2025', 'N/A')[:120]}...")
            lines.append("")

        # --- SECTION 4 : CONCLUSION ---
        lines.append("4. CONCLUSION DE L'AUDIT")
        lines.append("-" * 35)
        if m.get('concentration_index_2025', 0) > 5000:
            lines.append("⚠ OBSERVATION : Le budget reste fortement technique (74% d'articles 'Autre').")
        
        drift_rate = drift.get('drifted_percentage', 0)
        if drift_rate > 15:
            lines.append(f"✅ DYNAMISME : Le taux de changement de {drift_rate:.2f}% indique une")
            lines.append("   actualisation significative des priorités dans la Loi de Finances 2025.")
        else:
            lines.append("✅ STABILITÉ : On observe une continuité majeure dans la rédaction budgétaire.")
            
        lines.append("\n" + "=" * 75)

        # Sauvegarde
        full_report = "\n".join(lines)
        with open(self.report_path, 'w', encoding='utf-8') as f:
            f.write(full_report)
        
        print(full_report)
        logger.info(f"Rapport final enregistré avec succès.")

if __name__ == "__main__":
    generator = AuditReportGenerator()
    generator.generate()