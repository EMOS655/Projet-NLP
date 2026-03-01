import torch
from sentence_transformers import SentenceTransformer, util
import json
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score, log_loss, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

class SND30Classifier:
    def __init__(self):
        print("[#] Chargement du modèle expert (Version Spéciale Dé-noyage)...")
        # Utilisation du modèle multilingue performant
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 1. RÉGLAGE DES MOTS-CLÉS (Dictionnaire de référence métier)
        self.piliers_keywords = {
            "Transformation structurelle": [
                "industrie", "manufacturier", "usine", "agro-industrie", "transformation bois", 
                "filière coton", "exploitation minière", "fer", "or", "hydrocarbures", 
                "barrage", "centrale électrique", "pont", "autoroute", "port autonome", 
                "import-substitution", "made in cameroon", "technologie numérique", "automobile"
            ],
            "Capital humain": [
                "santé publique", "hôpital", "médicament", "vaccin", "couverture santé", 
                "maternité", "éducation", "école", "lycée", "université", "enseignement", 
                "formation professionnelle", "alphabétisation", "protection sociale", "emploi"
            ],
            "Gouvernance": [
                "justice", "tribunal", "corruption", "police", "gendarmerie", "armée", 
                "administration publique", "élection", "fiscalité", "douane", "impôts", "recettes"
            ],
            "Développement régional": [
                "décentralisation", "commune", "mairie", "conseil régional", "ctd", 
                "électrification rurale", "forage", "eau potable", "piste rurale"
            ],
            "Autre": [
                "dette publique", "amortissement", "fonctionnement", "salaire", "loyer", 
                "fourniture", "indemnité", "frais de justice", "imprévus"
            ]
        }
        
        self.category_names = list(self.piliers_keywords.keys())
        # Encodage préalable des mots-clés pour accélérer la classification
        self.encoded_piliers = {p: self.model.encode(k, convert_to_tensor=True) 
                               for p, k in self.piliers_keywords.items()}

    def classify(self, text):
        """ Classifie un texte en utilisant la similarité cosinus boostée """
        if not text or len(str(text)) < 15:
            return "Autre", 0.0, {p: 0.0 for p in self.category_names}

        # 2. NETTOYAGE DU BRUIT (Stop-words institutionnels)
        noise = ["ministre", "disposition", "alinéa", "loi", "cadre", "œuvre", "fixer", "relatif", "président", "république", "article"]
        clean_text = str(text).lower()
        for word in noise:
            clean_text = clean_text.replace(word, "")

        query_emb = self.model.encode(clean_text, convert_to_tensor=True)
        all_scores = {}

        # 3. CALCUL DU SCORE PAR INTENSITÉ (Moyenne des Top-2 mots-clés)
        for pilier, key_embs in self.encoded_piliers.items():
            sims = util.cos_sim(query_emb, key_embs)[0]
            top_v, _ = torch.topk(sims, k=min(2, len(sims)))
            all_scores[pilier] = torch.mean(top_v).item()

        # 4. LE BOOST STRATÉGIQUE (Ajustement empirique)
        all_scores["Transformation structurelle"] *= 1.25
        all_scores["Capital humain"] *= 1.25

        best_label = max(all_scores, key=all_scores.get)
        max_sim = all_scores[best_label]

        # Arbitrage Final (Seuil de confiance)
        if max_sim < 0.38:
            best_label = "Autre"

        return best_label, round(float(max_sim), 4), all_scores

    def get_performance_metrics(self, test_file_path):
        """ Calcule Accuracy, F1 et Log-Loss sur le set de test """
        print(f"[#] Évaluation de performance sur : {test_file_path}")
        if not Path(test_file_path).exists(): return None
        
        with open(test_file_path, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
            
        y_true, y_pred, y_probs = [], [], []
        labels_ordonnes = sorted(self.category_names)
        
        for group in test_data.get("test_set", []):
            label_reel = group.get("pilier")
            for art in group.get("articles", []):
                text_input = f"{art.get('titre', '')} {art.get('contenu', '')}"
                pred_label, _, all_scores = self.classify(text_input)
                
                y_true.append(label_reel)
                y_pred.append(pred_label)
                
                # Conversion des scores en probabilités (Softmax) pour le Log-Loss
                scores_bruts = np.array([all_scores[l] for l in labels_ordonnes])
                exp_scores = np.exp(scores_bruts - np.max(scores_bruts))
                probs = exp_scores / exp_scores.sum()
                y_probs.append(probs.tolist())
                
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_micro": round(f1_score(y_true, y_pred, average='micro'), 4),
            "log_loss": round(log_loss(y_true, y_probs, labels=labels_ordonnes), 4)
        }, y_true, y_probs

    def save_roc_data(self, y_true, y_probs, output_path="data/results/roc_curves.json"):
        """ Génère les données FPR/TPR pour la courbe ROC (One-vs-Rest) """
        labels_ordonnes = sorted(self.category_names)
        y_true_bin = label_binarize(y_true, classes=labels_ordonnes)
        y_probs = np.array(y_probs)
        
        roc_results = {}
        for i, label in enumerate(labels_ordonnes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_results[label] = {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
                "auc": round(auc(fpr, tpr), 4)
            }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(roc_results, f, ensure_ascii=False)
        print(f"[+] Courbes ROC sauvegardées dans : {output_path}")

    def run_audit(self, input_path, test_path):
        """ Lance l'audit complet et l'évaluation """
        if not os.path.exists(input_path):
            print(f"[!] Erreur : {input_path} introuvable.")
            return

        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Audit des articles
        for year in [y for y in data.keys() if y not in ["metrics", "SND30", "roc_data"]]:
            print(f"[#] Audit exercice {year}...")
            for art in data[year].get("articles", []):
                label, score, _ = self.classify(f"{art.get('titre_brut','')} {art.get('semantic_input','')}")
                art['label'] = label
                art['score'] = score
        
        # Évaluation et ROC
        metrics, y_true, y_probs = self.get_performance_metrics(test_path)
        data['metrics'] = metrics
        self.save_roc_data(y_true, y_probs)
        
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print("[SUCCESS] Audit et évaluation terminés avec succès.")

if __name__ == "__main__":
    clf = SND30Classifier()
    # Chemins à adapter selon votre structure
    clf.run_audit(
        input_path="data/results/all_processed_data.json",
        test_path="data/test/data_test_SND30_2026.json"
    )