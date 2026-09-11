import pytest
from src.evaluation.metrics import (
    compute_intent_metrics,
    compute_triage_metrics,
    compute_rouge_l,
    compute_reply_metrics
)
from src.evaluation.agreement_study import (
    compute_cohen_kappa,
    compute_correlation_and_agreement
)

def test_rouge_l_identical():
    s = "We are here for you. Check Settings > General > About."
    assert compute_rouge_l(s, s) == 1.0

def test_rouge_l_disjoint():
    s1 = "hello world"
    s2 = "goodbye earth"
    assert compute_rouge_l(s1, s2) == 0.0

def test_triage_metrics():
    y_true = ["ESCALATE", "AUTO_HANDLE", "ESCALATE", "AUTO_HANDLE"]
    y_pred = ["ESCALATE", "AUTO_HANDLE", "AUTO_HANDLE", "AUTO_HANDLE"]
    res = compute_triage_metrics(y_true, y_pred)
    assert res["triage_accuracy"] == 0.75
    assert res["critical_miss_rate"] == 0.50

def test_agreement_perfect():
    human = [1.0, 2.0, 3.0, 4.0, 5.0]
    judge = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = compute_correlation_and_agreement(human, judge)
    assert res["exact_agreement_rate"] == 1.0
    assert abs(res["pearson_r"] - 1.0) < 1e-4
