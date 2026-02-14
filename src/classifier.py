import torch
from sentence_transformers import SentenceTransformer, util
import json
from pathlib import Path
from sklearn.metrics import f1_score, log_loss, accuracy_score
import numpy as np

class SND30Classifier:
    def __init__(self):
        print(" Chargement du modèle expert (Version Spéciale Dé-noyage)...")
        self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        # 1. RÉGLAGE DES MOTS-CLÉS (Plus précis, moins de bruit)
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
        self.encoded_piliers = {p: self.model.encode(k, convert_to_tensor=True) 
                               for p, k in self.piliers_keywords.items()}

    def classify(self, text):
        if not text or len(str(text)) < 15:
            return "Autre", 0.0, {p: 0.0 for p in self.category_names}

        # 2. NETTOYAGE DU BRUIT (On enlève ce qui noie les classes)
        noise = ["ministre", "disposition", "alinéa", "loi", "cadre", "œuvre", "fixer", "relatif", "président", "république", "article"]
        clean_text = str(text).lower()
        for word in noise:
            clean_text = clean_text.replace(word, "")

        query_emb = self.model.encode(clean_text, convert_to_tensor=True)
        all_scores = {}

        # 3. CALCUL DU SCORE PAR INTENSITÉ (Top-2)
        for pilier, key_embs in self.encoded_piliers.items():
            sims = util.cos_sim(query_emb, key_embs)[0]
            # On prend les 2 meilleures correspondances
            top_v, _ = torch.topk(sims, k=min(2, len(sims)))
            all_scores[pilier] = torch.mean(top_v).item()

        # 4. LE BOOST (Pour faire ressortir TS et CH)
        # On donne un avantage de 20% aux deux premières classes
        all_scores["Transformation structurelle"] *= 1.25
        all_scores["Capital humain"] *= 1.25

        best_label = max(all_scores, key=all_scores.get)
        max_sim = all_scores[best_label]

        # Arbitrage Final
        if max_sim < 0.38:
            best_label = "Autre"

        return best_label, round(float(max_sim), 4), all_scores

    def evaluate_performance(self, test_file_path):
        print(f" Évaluation sur : {test_file_path}")
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
                scores_bruts = np.array([all_scores[l] for l in labels_ordonnes])
                exp_scores = np.exp(scores_bruts - np.max(scores_bruts))
                probs = exp_scores / exp_scores.sum()
                y_probs.append(probs.tolist())
        return {
            "accuracy": round(accuracy_score(y_true, y_pred), 4),
            "f1_micro": round(f1_score(y_true, y_pred, average='micro'), 4),
            "log_loss": round(log_loss(y_true, y_probs, labels=labels_ordonnes), 4)
        }

    def run_audit(self, input_path, test_path):
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for year in [y for y in data.keys() if y not in ["metrics", "SND30"]]:
            print(f" Audit exercice {year}...")
            for art in data[year].get("articles", []):
                label, score, _ = self.classify(f"{art.get('titre_brut','')} {art.get('semantic_input','')}")
                art['label'] = label
                art['score'] = score
        data['metrics'] = self.evaluate_performance(test_path)
        with open(input_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(" Audit terminé avec succès.")

if __name__ == "__main__":
    clf = SND30Classifier()
    clf.run_audit(
        input_path="data/results/all_processed_data.json",
        test_path="data/test/data_test_SND30_2026.json"
    )