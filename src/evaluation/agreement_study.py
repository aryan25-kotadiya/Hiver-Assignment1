import math
from typing import List, Dict, Any, Tuple
import numpy as np

def compute_cohen_kappa(human_scores: List[int], judge_scores: List[int], min_rating: int = 1, max_rating: int = 5) -> float:
    """
    Computes Cohen's linearly weighted kappa between human and judge scores.
    """
    n = len(human_scores)
    if n == 0:
        return 0.0

    k = max_rating - min_rating + 1
    matrix = np.zeros((k, k))
    for h, j in zip(human_scores, judge_scores):
        h_idx = max(0, min(k - 1, int(round(h)) - min_rating))
        j_idx = max(0, min(k - 1, int(round(j)) - min_rating))
        matrix[h_idx][j_idx] += 1

    # Marginal sums
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)

    # Linear weights
    weights = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            weights[i][j] = abs(i - j) / (k - 1)

    expected = np.outer(row_sums, col_sums) / n

    po = np.sum(weights * matrix) / n
    pe = np.sum(weights * expected) / n

    if pe == 1.0:
        return 1.0
    kappa = 1.0 - (po / pe)
    return float(kappa)

def compute_correlation_and_agreement(human_scores: List[float], judge_scores: List[float]) -> Dict[str, float]:
    """
    Calculates exact agreement, within-1 point agreement, Pearson r, and MAE.
    """
    n = len(human_scores)
    if n == 0:
        return {}

    exact_matches = sum(1 for h, j in zip(human_scores, judge_scores) if round(h) == round(j))
    within_one = sum(1 for h, j in zip(human_scores, judge_scores) if abs(round(h) - round(j)) <= 1)

    exact_agreement_rate = exact_matches / n
    within_one_rate = within_one / n

    mae = sum(abs(h - j) for h, j in zip(human_scores, judge_scores)) / n

    # Pearson correlation r
    h_mean = sum(human_scores) / n
    j_mean = sum(judge_scores) / n

    num = sum((h - h_mean) * (j - j_mean) for h, j in zip(human_scores, judge_scores))
    den_h = math.sqrt(sum((h - h_mean) ** 2 for h in human_scores))
    den_j = math.sqrt(sum((j - j_mean) ** 2 for j in judge_scores))

    pearson_r = (num / (den_h * den_j)) if (den_h * den_j) > 0 else 0.0

    # Kappa (on rounded integers)
    h_int = [int(round(x)) for x in human_scores]
    j_int = [int(round(x)) for x in judge_scores]
    kappa = compute_cohen_kappa(h_int, j_int)

    return {
        "sample_size": n,
        "exact_agreement_rate": float(exact_agreement_rate),
        "within_one_agreement_rate": float(within_one_rate),
        "pearson_r": float(pearson_r),
        "cohen_kappa": float(kappa),
        "mae": float(mae),
        "human_mean": float(h_mean),
        "judge_mean": float(j_mean)
    }
