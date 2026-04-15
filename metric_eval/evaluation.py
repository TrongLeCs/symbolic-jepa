from metrics.scoring_logic import LogicEvaluator
from metrics.scoring_semantic import SemanticEvaluator
from metrics.scoring_wellformedness import WellFormednessEvaluator
from evaluate import load
import nltk

class FOLEvaluator:
    def __init__(self):
        self.logic_evaluator = LogicEvaluator()
        self.semantic_evaluator = SemanticEvaluator()
        self.wellformedness_evaluator = WellFormednessEvaluator()

        self.bleu_evaluator = load("bleu")
        self.rouge_evaluator = load("rouge")

    def evaluate(self, fol_target, fol_predict):
        decoded_labels = [fol_target]
        decoded_preds = [fol_predict]

        wellformedness_score = self.wellformedness_evaluator.evaluate(fol_predict)["swf_final_score"]
        semantic_score = self.semantic_evaluator.evaluate(fol_target, fol_predict)
        logic_score = self.logic_evaluator.evaluate(fol_target, fol_predict)

        exact_match_count = sum([1 if pred.strip() == label.strip() else 0 for pred, label in zip(decoded_preds, decoded_labels)])
        exact_match_score = round(exact_match_count / len(decoded_preds), 6)

        rouge_output = self.rouge_evaluator.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        )

        rouge1_score = round(rouge_output["rouge1"], 4)
        rouge2_score = round(rouge_output["rouge2"], 4)
        rougeL_score = round(rouge_output["rougeL"], 4)
        rougeLsum_score = round(rouge_output["rougeLsum"], 4)

        bleu_scores = self.bleu_evaluator.compute(predictions=decoded_preds, references=decoded_labels)
        bleu_score = bleu_scores['bleu']

        top1_count = 0
        for i in range(len(decoded_preds)):
            pred = nltk.sent_tokenize(decoded_preds[i].strip())
            label = nltk.sent_tokenize(decoded_labels[i].strip())
            if pred == label:
                top1_count += 1
        top1_score = round(top1_count / len(decoded_preds), 6)

        metrics_result = {
            "top-1 accuracy": top1_score,
            "bleu_score": bleu_score,
            "rouge1": rouge1_score,
            "rouge2": rouge2_score,
            "rougeL": rougeL_score,
            "rougeLsum": rougeLsum_score,
            "exact_match": exact_match_score,
            "well_formedness": wellformedness_score,
            "semantic_score": semantic_score,
            "logic_score": logic_score
        }

        return metrics_result