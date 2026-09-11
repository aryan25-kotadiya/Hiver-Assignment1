import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np

from src.config import INTENTS, ACTION_ESCALATE, ACTION_AUTO_HANDLE, MAX_TWEET_LENGTH
from src.nlp_engine import compute_metrics_helper

def compute_intent_metrics(true_intents: List[str], pred_intents: List[str]) -> Dict[str, float]:
    """
    Computes multi-class classification metrics across the 7 intents.
    """
    res = compute_metrics_helper(true_intents, pred_intents, labels=INTENTS)
    per_intent = {f"f1_{intent}": f1 for intent, f1 in res["per_class_f1"].items()}

    return {
        "intent_accuracy": res["accuracy"],
        "intent_macro_precision": res["macro_precision"],
        "intent_macro_recall": res["macro_recall"],
        "intent_macro_f1": res["macro_f1"],
        "intent_weighted_f1": res["weighted_f1"],
        **per_intent
    }

def compute_triage_metrics(true_actions: List[str], pred_actions: List[str]) -> Dict[str, float]:
    """
    Computes binary triage metrics focusing heavily on critical safety and cost trade-offs.
    """
    tp = sum(1 for t, p in zip(true_actions, pred_actions) if t == ACTION_ESCALATE and p == ACTION_ESCALATE)
    fp = sum(1 for t, p in zip(true_actions, pred_actions) if t == ACTION_AUTO_HANDLE and p == ACTION_ESCALATE)
    fn = sum(1 for t, p in zip(true_actions, pred_actions) if t == ACTION_ESCALATE and p == ACTION_AUTO_HANDLE)
    tn = sum(1 for t, p in zip(true_actions, pred_actions) if t == ACTION_AUTO_HANDLE and p == ACTION_AUTO_HANDLE)

    total_escalations = tp + fn
    total_autohandles = tn + fp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / total_escalations if total_escalations > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    critical_miss_rate = fn / total_escalations if total_escalations > 0 else 0.0
    false_escalation_rate = fp / total_autohandles if total_autohandles > 0 else 0.0
    accuracy = (tp + tn) / len(true_actions) if len(true_actions) > 0 else 0.0

    return {
        "triage_accuracy": float(accuracy),
        "escalation_precision": float(precision),
        "escalation_recall": float(recall),
        "escalation_f1": float(f1),
        "critical_miss_rate": float(critical_miss_rate),
        "false_escalation_rate": float(false_escalation_rate),
        "raw_counts": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    }

def compute_rouge_l(reference: str, candidate: str) -> float:
    """
    Calculates ROUGE-L (Longest Common Subsequence F1) between strings.
    """
    ref_tokens = re.findall(r"\w+", reference.lower())
    cand_tokens = re.findall(r"\w+", candidate.lower())

    if not ref_tokens or not cand_tokens:
        return 0.0

    m, n = len(ref_tokens), len(cand_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == cand_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    prec = lcs_len / n
    rec = lcs_len / m
    if prec + rec == 0:
        return 0.0
    return (2 * prec * rec) / (prec + rec)

def compute_reply_metrics(references: List[str], candidates: List[str]) -> Dict[str, float]:
    """
    Computes linguistic quality and brand constraint metrics.
    """
    rouge_l_scores = [compute_rouge_l(ref, cand) for ref, cand in zip(references, candidates)]
    mean_rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) if rouge_l_scores else 0.0

    length_violations = sum(1 for c in candidates if len(c) > MAX_TWEET_LENGTH)
    length_compliance_rate = 1.0 - (length_violations / len(candidates)) if candidates else 1.0

    pii_violations = sum(
        1 for c in candidates
        if re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|serial\s*#?\s*[A-Z0-9]{10}", c, re.I)
    )
    pii_safety_rate = 1.0 - (pii_violations / len(candidates)) if candidates else 1.0

    return {
        "rouge_l": float(mean_rouge_l),
        "length_compliance_rate": float(length_compliance_rate),
        "pii_safety_rate": float(pii_safety_rate)
    }
