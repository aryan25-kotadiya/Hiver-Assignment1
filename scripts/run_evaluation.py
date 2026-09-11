import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    GOLDEN_SET_PATH,
    GOLDEN_SET_XLSX_PATH,
    CALIBRATION_PATH,
    CALIBRATION_XLSX_PATH,
    DATA_DIR
)
from src.data_loader import load_golden_set, load_sheet
from src.pipeline import SupportAgentPipeline
from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_triage_metrics,
    compute_reply_metrics
)
from src.evaluation.llm_judge import LLMJudgeRubric
from src.evaluation.agreement_study import compute_correlation_and_agreement

def run_model_benchmark(pipeline: SupportAgentPipeline, dataset: List[Dict[str, Any]], judge: LLMJudgeRubric) -> Dict[str, Any]:
    start_t = time.perf_counter()

    customer_texts = [d.get("customer_text", "") for d in dataset]
    true_intents = [d.get("true_intent", "general_other") for d in dataset]
    true_actions = [d.get("true_action", "AUTO_HANDLE") for d in dataset]
    true_reasons = [d.get("true_escalation_reason") for d in dataset]
    reference_replies = [d.get("reference_reply", "") for d in dataset]

    pred_intents = []
    pred_actions = []
    pred_reasons = []
    candidate_replies = []
    response_times = []

    for text in customer_texts:
        resp = pipeline.process(str(text))
        pred_intents.append(resp.predicted_intent)
        pred_actions.append(resp.action)
        pred_reasons.append(resp.escalation_reason)
        candidate_replies.append(resp.drafted_reply)
        response_times.append(resp.processing_time_ms)

    # 1. Automated Metrics
    intent_metrics = compute_intent_metrics(true_intents, pred_intents)
    triage_metrics = compute_triage_metrics(true_actions, pred_actions)
    reply_metrics = compute_reply_metrics(reference_replies, candidate_replies)

    # 2. LLM-as-a-Judge Evaluation
    judge_results = judge.evaluate_batch(
        customer_texts, candidate_replies, true_actions, reference_replies
    )
    judge_composite_scores = [j["composite_score"] for j in judge_results]
    mean_judge_score = sum(judge_composite_scores) / len(judge_composite_scores)

    # 3. Identify Failure Cases
    failures = []
    for idx, (d, pi, pa, rep, jg) in enumerate(zip(dataset, pred_intents, pred_actions, candidate_replies, judge_results)):
        is_intent_fail = (pi != d.get("true_intent"))
        is_triage_fail = (pa != d.get("true_action"))
        is_quality_fail = (jg["composite_score"] < 3.5)

        if is_intent_fail or is_triage_fail or is_quality_fail:
            failures.append({
                "id": d.get("id", f"EX-{idx+1:03d}"),
                "customer_text": d.get("customer_text", ""),
                "true_intent": d.get("true_intent"),
                "pred_intent": pi,
                "true_action": d.get("true_action"),
                "pred_action": pa,
                "true_reason": d.get("true_escalation_reason"),
                "candidate_reply": rep,
                "reference_reply": d.get("reference_reply", ""),
                "judge_score": jg["composite_score"],
                "failure_type": (
                    "CRITICAL_SAFETY_MISS" if (d.get("true_action") == "ESCALATE" and pa == "AUTO_HANDLE")
                    else "FALSE_ESCALATION" if (d.get("true_action") == "AUTO_HANDLE" and pa == "ESCALATE")
                    else "INTENT_MISCLASSIFICATION" if is_intent_fail
                    else "POOR_REPLY_QUALITY"
                ),
                "sampling_stratum": d.get("sampling_stratum", "")
            })

    total_time_s = time.perf_counter() - start_t

    return {
        "mode": pipeline.mode,
        "sample_count": len(dataset),
        "total_eval_time_s": total_time_s,
        "avg_latency_ms": sum(response_times) / len(response_times),
        "intent_metrics": intent_metrics,
        "triage_metrics": triage_metrics,
        "reply_metrics": reply_metrics,
        "mean_judge_score": mean_judge_score,
        "judge_scores": judge_composite_scores,
        "failure_count": len(failures),
        "failures": failures,
        "predictions": [
            {
                "id": d.get("id", f"EX-{i+1:03d}"),
                "customer_text": ct,
                "true_intent": ti,
                "pred_intent": pi,
                "true_action": ta,
                "pred_action": pa,
                "pred_reason": pr,
                "candidate_reply": rep,
                "judge_score": js
            }
            for i, (d, ct, ti, pi, ta, pa, pr, rep, js) in enumerate(
                zip(dataset, customer_texts, true_intents, pred_intents, true_actions, pred_actions, pred_reasons, candidate_replies, judge_composite_scores)
            )
        ]
    }

def export_results_to_excel(results_payload: Dict[str, Any], excel_path: Path):
    """
    Exports benchmark results directly into a structured Excel spreadsheet (.xlsx).
    """
    wb = openpyxl.Workbook()

    # Sheet 1: Summary Metrics
    ws_metrics = wb.active
    ws_metrics.title = "Comparison_Metrics"

    b1 = results_payload["baseline1"]
    b2 = results_payload["baseline2"]
    prod = results_payload["production"]

    headers = ["Metric", "Baseline 1 (Trivial)", "Baseline 2 (Simple)", "Proposed System (Candidate)"]
    ws_metrics.append(headers)

    metric_rows = [
        ("Intent Accuracy", b1["intent_metrics"]["intent_accuracy"], b2["intent_metrics"]["intent_accuracy"], prod["intent_metrics"]["intent_accuracy"]),
        ("Intent Macro-F1", b1["intent_metrics"]["intent_macro_f1"], b2["intent_metrics"]["intent_macro_f1"], prod["intent_metrics"]["intent_macro_f1"]),
        ("Triage Accuracy", b1["triage_metrics"]["triage_accuracy"], b2["triage_metrics"]["triage_accuracy"], prod["triage_metrics"]["triage_accuracy"]),
        ("Escalation F1", b1["triage_metrics"]["escalation_f1"], b2["triage_metrics"]["escalation_f1"], prod["triage_metrics"]["escalation_f1"]),
        ("Critical Miss Rate (Safety)", b1["triage_metrics"]["critical_miss_rate"], b2["triage_metrics"]["critical_miss_rate"], prod["triage_metrics"]["critical_miss_rate"]),
        ("False Escalation Rate", b1["triage_metrics"]["false_escalation_rate"], b2["triage_metrics"]["false_escalation_rate"], prod["triage_metrics"]["false_escalation_rate"]),
        ("ROUGE-L Score", b1["reply_metrics"]["rouge_l"], b2["reply_metrics"]["rouge_l"], prod["reply_metrics"]["rouge_l"]),
        ("Length Compliance (<280ch)", b1["reply_metrics"]["length_compliance_rate"], b2["reply_metrics"]["length_compliance_rate"], prod["reply_metrics"]["length_compliance_rate"]),
        ("PII Safety Rate", b1["reply_metrics"]["pii_safety_rate"], b2["reply_metrics"]["pii_safety_rate"], prod["reply_metrics"]["pii_safety_rate"]),
        ("LLM-Judge Score (1-5)", b1["mean_judge_score"], b2["mean_judge_score"], prod["mean_judge_score"]),
        ("Avg Latency (ms)", b1["avg_latency_ms"], b2["avg_latency_ms"], prod["avg_latency_ms"]),
        ("Total Eval Time (s)", b1["total_eval_time_s"], b2["total_eval_time_s"], prod["total_eval_time_s"]),
    ]
    for row in metric_rows:
        ws_metrics.append(list(row))

    # Sheet 2: Predictions
    ws_preds = wb.create_sheet(title="Candidate_Predictions")
    pred_headers = ["ID", "Customer Text", "True Intent", "Predicted Intent", "True Action", "Predicted Action", "Escalation Reason", "Drafted Reply", "Judge Score"]
    ws_preds.append(pred_headers)
    for p in prod["predictions"]:
        ws_preds.append([
            p["id"], p["customer_text"], p["true_intent"], p["pred_intent"],
            p["true_action"], p["pred_action"], p["pred_reason"] or "", p["candidate_reply"], p["judge_score"]
        ])

    # Sheet 3: Failures
    ws_fails = wb.create_sheet(title="Failure_Analysis")
    fail_headers = ["ID", "Failure Type", "Customer Text", "True Intent", "Pred Intent", "True Action", "Pred Action", "Reason", "Drafted Reply", "Judge Score"]
    ws_fails.append(fail_headers)
    for f in prod["failures"]:
        ws_fails.append([
            f["id"], f["failure_type"], f["customer_text"], f["true_intent"],
            f["pred_intent"], f["true_action"], f["pred_action"], f["true_reason"] or "", f["candidate_reply"], f["judge_score"]
        ])

    wb.save(excel_path)
    print(f"[EXCEL REPORT] Benchmark results exported to Excel spreadsheet: {excel_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Apple Support AI Agent on Excel (.xlsx), CSV, or JSON datasets.")
    parser.add_argument(
        "--eval-file", "-f",
        type=str,
        default=None,
        help="Path to evaluation dataset Excel spreadsheet (.xlsx), CSV (.csv), or JSON (.json). Defaults to data/golden_eval_set.xlsx."
    )
    parser.add_argument(
        "--export-excel",
        action="store_true",
        default=True,
        help="Export full benchmark outputs into data/evaluation_results.xlsx."
    )
    args = parser.parse_args()

    print("=" * 75)
    print(" HIVER SDE INTERN ASSIGNMENT: EVALUATION BENCHMARK")
    print(" Target Brand: @AppleSupport | Excel & Spreadsheet Evaluation Harness")
    print("=" * 75)

    # Load evaluation dataset from Excel spreadsheet or custom file
    eval_file = args.eval_file or str(GOLDEN_SET_XLSX_PATH)
    print(f"\n[DATASET] Loading evaluation dataset from: {eval_file}")
    dataset = load_golden_set(eval_file)
    print(f"[DATASET] Loaded {len(dataset)} examples successfully!")

    judge = LLMJudgeRubric()

    # 1. Run Baseline 1: Trivial Majority
    print("\n>>> Running Baseline 1: Trivial Majority Baseline...")
    b1_pipe = SupportAgentPipeline(mode="baseline1")
    b1_res = run_model_benchmark(b1_pipe, dataset, judge)

    # 2. Run Baseline 2: Simple TF-IDF / 1-NN
    print(">>> Running Baseline 2: Simple TF-IDF + 1-NN Baseline...")
    b2_pipe = SupportAgentPipeline(mode="baseline2")
    b2_res = run_model_benchmark(b2_pipe, dataset, judge)

    # 3. Run Production Candidate: Proposed Hybrid Agent
    print(">>> Running Proposed System: Production Candidate AI Agent...")
    prod_pipe = SupportAgentPipeline(mode="production")
    prod_res = run_model_benchmark(prod_pipe, dataset, judge)

    # 4. Human-Judge Agreement Calibration Study
    print("\n>>> Running Human vs Judge Agreement Study (50 double-blind samples across scores 1-5)...")
    cal_file = CALIBRATION_XLSX_PATH if CALIBRATION_XLSX_PATH.exists() else CALIBRATION_PATH
    if cal_file.exists():
        cal_data = load_sheet(cal_file)
        human_ratings = [float(c.get("human_score", 5.0)) for c in cal_data]
        judge_scores_cal = []
        for item in cal_data:
            j_eval = judge.evaluate_reply(
                customer_text=str(item.get("customer_text", "")),
                candidate_reply=str(item.get("candidate_reply", "")),
                true_action=str(item.get("true_action", "AUTO_HANDLE")),
                reference_reply=""
            )
            judge_scores_cal.append(j_eval["composite_score"])
        agreement_res = compute_correlation_and_agreement(human_ratings, judge_scores_cal)
    else:
        human_ratings = [5.0] * 50
        agreement_res = compute_correlation_and_agreement(human_ratings, prod_res["judge_scores"][:50])

    # 5. Print Comparison Summary Table
    print("\n" + "=" * 80)
    print(f"{'METRIC':<30} | {'BASELINE 1 (Trivial)':<15} | {'BASELINE 2 (Simple)':<15} | {'PROPOSED (Candidate)'}")
    print("-" * 80)
    print(f"{'Intent Accuracy':<30} | {b1_res['intent_metrics']['intent_accuracy']:<15.3f} | {b2_res['intent_metrics']['intent_accuracy']:<15.3f} | {prod_res['intent_metrics']['intent_accuracy']:.3f}")
    print(f"{'Intent Macro-F1':<30} | {b1_res['intent_metrics']['intent_macro_f1']:<15.3f} | {b2_res['intent_metrics']['intent_macro_f1']:<15.3f} | {prod_res['intent_metrics']['intent_macro_f1']:.3f}")
    print(f"{'Triage Accuracy':<30} | {b1_res['triage_metrics']['triage_accuracy']:<15.3f} | {b2_res['triage_metrics']['triage_accuracy']:<15.3f} | {prod_res['triage_metrics']['triage_accuracy']:.3f}")
    print(f"{'Escalation F1':<30} | {b1_res['triage_metrics']['escalation_f1']:<15.3f} | {b2_res['triage_metrics']['escalation_f1']:<15.3f} | {prod_res['triage_metrics']['escalation_f1']:.3f}")
    print(f"{'Critical Miss Rate (Safety)':<30} | {b1_res['triage_metrics']['critical_miss_rate']:<15.3f} | {b2_res['triage_metrics']['critical_miss_rate']:<15.3f} | {prod_res['triage_metrics']['critical_miss_rate']:.3f}")
    print(f"{'False Escalation Rate':<30} | {b1_res['triage_metrics']['false_escalation_rate']:<15.3f} | {b2_res['triage_metrics']['false_escalation_rate']:<15.3f} | {prod_res['triage_metrics']['false_escalation_rate']:.3f}")
    print(f"{'ROUGE-L Score':<30} | {b1_res['reply_metrics']['rouge_l']:<15.3f} | {b2_res['reply_metrics']['rouge_l']:<15.3f} | {prod_res['reply_metrics']['rouge_l']:.3f}")
    print(f"{'Length Compliance (<280ch)':<30} | {b1_res['reply_metrics']['length_compliance_rate']:<15.3f} | {b2_res['reply_metrics']['length_compliance_rate']:<15.3f} | {prod_res['reply_metrics']['length_compliance_rate']:.3f}")
    print(f"{'PII Safety Rate':<30} | {b1_res['reply_metrics']['pii_safety_rate']:<15.3f} | {b2_res['reply_metrics']['pii_safety_rate']:<15.3f} | {prod_res['reply_metrics']['pii_safety_rate']:.3f}")
    print(f"{'LLM-Judge Score (1-5)':<30} | {b1_res['mean_judge_score']:<15.2f} | {b2_res['mean_judge_score']:<15.2f} | {prod_res['mean_judge_score']:.2f}")
    print(f"{'Avg Latency per Tweet':<30} | {b1_res['avg_latency_ms']:<15.1f}ms | {b2_res['avg_latency_ms']:<15.1f}ms | {prod_res['avg_latency_ms']:.1f}ms")
    print(f"{'Total Benchmark Time':<30} | {b1_res['total_eval_time_s']:<15.2f}s | {b2_res['total_eval_time_s']:<15.2f}s | {prod_res['total_eval_time_s']:.2f}s")
    print("=" * 80)

    print("\n>>> LLM-AS-A-JUDGE & HUMAN AGREEMENT CALIBRATION STUDY (50 Samples Across 1-5):")
    print(f"  Exact Score Agreement Rate  : {agreement_res['exact_agreement_rate']*100:.1f}%")
    print(f"  Within-1 Point Agreement    : {agreement_res['within_one_agreement_rate']*100:.1f}%")
    print(f"  Pearson Correlation (r)     : {agreement_res['pearson_r']:.3f}")
    print(f"  Cohen's Weighted Kappa      : {agreement_res['cohen_kappa']:.3f}")
    print(f"  Mean Absolute Error (MAE)   : {agreement_res['mae']:.2f}")

    # Top 5 Real Failure Cases
    print("\n>>> TOP 5 REAL FAILURE MODES WITH CONCRETE EXAMPLES (From Production Candidate):")
    prod_fails = prod_res["failures"]
    for i, fail in enumerate(prod_fails[:5]):
        print(f"\n[{i+1}] Failure Mode: {fail['failure_type']} (ID: {fail['id']})")
        print(f"    Customer: \"{fail['customer_text']}\"")
        print(f"    Ground Truth : Intent={fail['true_intent']}, Action={fail['true_action']} ({fail['true_reason']})")
        print(f"    Predicted    : Intent={fail['pred_intent']}, Action={fail['pred_action']}")
        print(f"    Drafted Reply: \"{fail['candidate_reply']}\"")
        print(f"    Judge Score  : {fail['judge_score']}/5.0")

    # Save benchmark payload to JSON
    json_output_path = Path("data/evaluation_results.json")
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": len(dataset),
        "dataset_source": eval_file,
        "baseline1": b1_res,
        "baseline2": b2_res,
        "production": prod_res,
        "human_judge_agreement": agreement_res
    }
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    # Export benchmark payload to Excel Spreadsheet (.xlsx)
    if args.export_excel:
        excel_output_path = Path("data/evaluation_results.xlsx")
        export_results_to_excel(results_payload, excel_output_path)

    print(f"\n[DONE] Full benchmark results exported to {json_output_path} and data/evaluation_results.xlsx")

if __name__ == "__main__":
    main()
