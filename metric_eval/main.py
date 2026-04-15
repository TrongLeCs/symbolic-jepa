import argparse
import json
import os
from pathlib import Path
from statistics import mean
import nltk

from metrics.scoring_logic import LogicEvaluator
from metrics.scoring_semantic import SemanticEvaluator
from metrics.scoring_wellformedness import WellFormednessEvaluator
from evaluate import load


ARTIFACT_ENV = "LOGIC_JEPA_ARTIFACTS_DIR"


def _artifact_default(local_path: str, artifact_path: str) -> str:
    root = os.getenv(ARTIFACT_ENV, "").strip()
    if not root:
        return local_path
    return str(Path(root) / artifact_path)

# +++ helper
def harmonic_mean(a: float, b: float, eps: float = 1e-12) -> float:
    s = a + b
    if s <= eps:
        return 0.0
    return (2.0 * a * b) / s

class FOLEvaluator:
    def __init__(self):
        self.logic_evaluator = LogicEvaluator()
        self.semantic_evaluator = SemanticEvaluator()
        self.wellformedness_evaluator = WellFormednessEvaluator()
        self.bleu_evaluator = load("bleu")
        self.rouge_evaluator = load("rouge")

        # Ensure NLTK has a tokenizer
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception:
                pass

    def evaluate(self, fol_target, fol_predict):
        decoded_labels = [fol_target]
        decoded_preds = [fol_predict]

        wf_score = self.wellformedness_evaluator.evaluate(fol_predict)[
            "swf_final_score"
        ]
        sem_score = self.semantic_evaluator.evaluate(fol_target, fol_predict)
        log_score = self.logic_evaluator.evaluate(fol_target, fol_predict)

        exact_match = 1.0 if fol_predict.strip() == fol_target.strip() else 0.0

        rouge_output = self.rouge_evaluator.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            rouge_types=["rouge1", "rouge2", "rougeL", "rougeLsum"],
        )
        rouge1 = round(rouge_output["rouge1"], 4)
        rouge2 = round(rouge_output["rouge2"], 4)
        rougeL = round(rouge_output["rougeL"], 4)
        rougeLsum = round(rouge_output["rougeLsum"], 4)

        bleu = self.bleu_evaluator.compute(
            predictions=decoded_preds, references=decoded_labels
        )["bleu"]

        pred_sents = nltk.sent_tokenize(decoded_preds[0].strip())
        label_sents = nltk.sent_tokenize(decoded_labels[0].strip())
        top1 = 1.0 if pred_sents == label_sents else 0.0

        return {
            "accuracy": round(top1, 6),
            "bleu_score": bleu,
            "rouge1": rouge1,
            "rouge2": rouge2,
            "rougeL": rougeL,
            "rougeLsum": rougeLsum,
            "exact_match": round(exact_match, 6),
            "SWF": wf_score,
            "PSE": sem_score,
            "LE": log_score,
        }


def read_dataset(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Dataset must be a LIST of JSON objects.")
    return data


from tqdm import tqdm


def evaluate_dataset(
    input_path: Path,
    output_path: Path,
    lambda1: float = 0.7,
    lambda2: float = 0.3,
):
    # +++ validate lambdas
    if not (0.0 <= lambda1 <= 1.0 and 0.0 <= lambda2 <= 1.0):
        raise ValueError("lambda1 and lambda2 must be in [0,1].")
    if abs((lambda1 + lambda2) - 1.0) > 1e-9:
        raise ValueError("lambda1 + lambda2 must = 1.")

    data = read_dataset(input_path)
    evaluator = FOLEvaluator()

    TARGET_FIELD = "FOL"
    PRED_LIST_FIELD = "Predict-FOL"

    agg_keys = [
        "accuracy",
        "bleu_score",
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
        "exact_match",
        "SWF",
        "PSE",
        "LE",
        "CS"  # +++ add CS to aggregation
    ]
    agg_vals = {k: [] for k in agg_keys}
    details = []

    for idx, ex in enumerate(tqdm(data, desc="Evaluating", unit="sample")):
        nl = ex.get("NL", "")
        fol_target = (ex.get(TARGET_FIELD) or "").strip()
        preds = ex.get(PRED_LIST_FIELD, [])
        if isinstance(preds, str):
            preds = [preds]
        if not isinstance(preds, list):
            preds = []

        item = {
            "index": idx,
            "NL": nl,
            "Target-FOL": fol_target,
            "Predict-FOL": preds,
            "results-metric": {},
        }

        if preds:
            top1_pred = preds[0].strip()
            top1_metrics = evaluator.evaluate(fol_target, top1_pred)

            wf = float(top1_metrics["SWF"])
            lg = float(top1_metrics["LE"])
            sem = float(top1_metrics["PSE"])

            hm_wf_lg = harmonic_mean(wf, lg)
            final_score = lambda1 * hm_wf_lg + lambda2 * sem
            top1_metrics["CS"] = round(final_score, 6)
            item["results-metric"] = top1_metrics

            for k in agg_keys:
                agg_vals[k].append(top1_metrics[k])

        details.append(item)

    # --- average metrics (used internally only, not exported to file)
    avg_metrics = {k: (mean(v) if v else 0.0) for k, v in agg_vals.items()}

    # +++ final_score based on AVERAGE
    avg_wf = float(avg_metrics["SWF"])
    avg_lg = float(avg_metrics["LE"])
    avg_sem = float(avg_metrics["PSE"])
    hm_wf_lg = harmonic_mean(avg_wf, avg_lg)
    final_score = lambda1 * hm_wf_lg + lambda2 * avg_sem

    # +++ add to aggregate
    aggregate = {
        "SWF": avg_wf,
        "LE": avg_lg,
        "PSE": avg_sem,
        "CS": final_score,
    }

    out = {
        "num_samples": len(details),
        "lambdas": {"lambda1": lambda1, "lambda2": lambda2},
        "aggregate": aggregate,
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out

def main():
    parser = argparse.ArgumentParser(description="Evaluate Predict-FOL against FOL.")
    parser.add_argument(
        "--input",
        type=str,
        default=_artifact_default(
            "results_inference/test_00.json",
            "decoder/inference_results/test.json",
        ),
        help="Path to JSON file (array of objects).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=_artifact_default(
            "results_metrics/test.json",
            "metric_eval/results_metrics/test.json",
        ),
        help="Path to the result JSON file.",
    )
    # +++ weights
    parser.add_argument("--lambda1", type=float, default=0.5, help="Weight for harmonic mean of well-formedness and logic.")
    parser.add_argument("--lambda2", type=float, default=0.5, help="Weight for semantic score (sum must = 1).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    result = evaluate_dataset(
        input_path,
        output_path,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
    )
    print(f"Evaluated {result['num_samples']} samples. Results saved at: {output_path}")


if __name__ == "__main__":
    main()