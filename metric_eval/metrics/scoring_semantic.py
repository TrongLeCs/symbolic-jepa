import json
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util

class SemanticEvaluator:
    def __init__(self, _method='tfidf'):
        self._method = _method

        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_predicates(self, fol_expr: str) -> list:
        """Extract predicate names from the FOL expression."""
        preds = re.findall(r'([A-Za-z_][A-Za-z0-9_]*)\s*\(', fol_expr)
        predi = [p for p in preds if not (len(p) == 1 and p.islower())]

        # Phat added: remove _
        predicates = [p.replace('_', ' ') for p in predi]
        return predicates

    def compute_semantic_score_tfidf(self, target_fol: str, predicted_fol: str) -> float:
        target_preds = self.extract_predicates(target_fol)
        predict_preds = self.extract_predicates(predicted_fol)

        if not target_preds and not predict_preds:
            return 1.0
        if not target_preds or not predict_preds:
            return 0.0

        all_preds = list(set(target_preds + predict_preds))
        vectorizer = TfidfVectorizer().fit(all_preds)
        target_vecs = vectorizer.transform(target_preds).toarray()
        predict_vecs = vectorizer.transform(predict_preds).toarray()

        cosine_matrix = cosine_similarity(target_vecs, predict_vecs)

        return self.compute_predicate_alignment_score(cosine_matrix, target_preds, predict_preds)

    def compute_semantic_score_sbert(self, target_fol: str, predicted_fol: str) -> float:
        if self.sbert_model is None:
            raise ImportError("sentence-transformers is not installed. Please `pip install sentence-transformers`.")

        target_preds = self.extract_predicates(target_fol)
        predict_preds = self.extract_predicates(predicted_fol)

        if not target_preds and not predict_preds:
            return 1.0
        if not target_preds or not predict_preds:
            return 0.0

        target_embs = self.sbert_model.encode(target_preds, convert_to_tensor=True)
        predict_embs = self.sbert_model.encode(predict_preds, convert_to_tensor=True)

        cosine_matrix = util.cos_sim(target_embs, predict_embs).cpu().numpy()

        return self.compute_predicate_alignment_score(cosine_matrix, target_preds, predict_preds)

    def compute_predicate_alignment_score(self, cosine_matrix, target_preds, predict_preds) -> float:
        # Pass 1: target -> predict
        target_best_matches = {i: np.argmax(row) for i, row in enumerate(cosine_matrix)}

        # Pass 2: predict -> target
        predict_best_matches = {j: np.argmax(col) for j, col in enumerate(cosine_matrix.T)}

        union = []
        not_joint = []
        matched_target = set()
        matched_predict = set()

        for i, j in target_best_matches.items():
            if predict_best_matches.get(j) == i:
                sim_val = float(cosine_matrix[i][j])
                union.append((target_preds[i], predict_preds[j], sim_val))
                matched_target.add(i)
                matched_predict.add(j)

        for i in range(len(target_preds)):
            if i not in matched_target:
                not_joint.append((target_preds[i], None, 0.0))

        for j in range(len(predict_preds)):
            if j not in matched_predict:
                not_joint.append((None, predict_preds[j], 0.0))

        total_sim = sum(float(sim) for _, _, sim in union + not_joint)
        total_count = len(union) + len(not_joint)

        return float(total_sim / total_count) if total_count > 0 else 0.0

    def evaluate(self, fol_target, fol_predict):
        compute_fn = self.compute_semantic_score_sbert if self._method == "sbert" else self.compute_semantic_score_tfidf

        try:
            score = compute_fn(fol_target, fol_predict)
            score = float(min(max(score, 0.0), 1.0)) 
        except Exception as e:
            score = 0.0

        return score