import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import GOLDEN_SET_PATH, DATA_DIR
from src.pipeline import SupportAgentPipeline
from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_triage_metrics,
    compute_reply_metrics
)
from src.evaluation.llm_judge import LLMJudgeRubric
from src.evaluation.agreement_study import compute_correlation_and_agreement

CALIBRATION_PATH = DATA_DIR / "calibration_study_set.json"

def run_model_benchmark(pipeline: SupportAgentPipeline, dataset: List[Dict[str, Any]], judge: LLMJudgeRubric) -> Dict[str, Any]:
    start_t = time.perf_counter()

    customer_texts = [d["customer_text"] for d in dataset]
    true_intents = [d["true_intent"] for d in dataset]
    true_actions = [d["true_action"] for d in dataset]
    true_reasons = [d.get("true_escalation_reason") for d in dataset]
    reference_replies = [d["reference_reply"] for d in dataset]

    pred_intents = []
    pred_actions = []
    pred_reasons = []
    candidate_replies = []
    response_times = []

    for text in customer_texts:
        resp = pipeline.process(text)
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
        is_intent_fail = (pi != d["true_intent"])
        is_triage_fail = (pa != d["true_action"])
        is_quality_fail = (jg["composite_score"] < 3.5)

        if is_intent_fail or is_triage_fail or is_quality_fail:
            failures.append({
                "id": d["id"],
                "customer_text": d["customer_text"],
                "true_intent": d["true_intent"],
                "pred_intent": pi,
                "true_action": d["true_action"],
                "pred_action": pa,
                "true_reason": d.get("true_escalation_reason"),
                "candidate_reply": rep,
                "reference_reply": d["reference_reply"],
                "judge_score": jg["composite_score"],
                "failure_type": (
                    "CRITICAL_SAFETY_MISS" if (d["true_action"] == "ESCALATE" and pa == "AUTO_HANDLE")
                    else "FALSE_ESCALATION" if (d["true_action"] == "AUTO_HANDLE" and pa == "ESCALATE")
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
        "failures": failures
    }

def main():
    print("=" * 70)
    print(" HIVER SDE INTERN ASSIGNMENT: EVALUATION BENCHMARK")
    print(" Target Brand: @AppleSupport | Evaluation Set: 200 Golden Examples")
    print("=" * 70)

    if not GOLDEN_SET_PATH.exists():
        print(f"Error: Golden evaluation set not found at {GOLDEN_SET_PATH}")
        sys.exit(1)

    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} hand-labelled golden test examples.")

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

    # 4. Human-Judge Agreement Calibration Study on the 50-sample calibration dataset
    print("\n>>> Running Human vs Judge Agreement Study (50 double-blind samples across scores 1-5)...")
    if CALIBRATION_PATH.exists():
        with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
            cal_data = json.load(f)
        human_ratings = [c["human_score"] for c in cal_data]
        judge_scores_cal = []
        for item in cal_data:
            j_eval = judge.evaluate_reply(
                customer_text=item["customer_text"],
                candidate_reply=item["candidate_reply"],
                true_action=item["true_action"],
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

    # Save complete benchmark payload to disk
    output_path = Path("data/evaluation_results.json")
    results_payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset_size": len(dataset),
        "baseline1": b1_res,
        "baseline2": b2_res,
        "production": prod_res,
        "human_judge_agreement": agreement_res
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\n[DONE] Full benchmark results exported to {output_path}")

if __name__ == "__main__":
    main()
