import re
import math
from typing import List, Dict, Any, Tuple, Optional
from collections import Counter
import numpy as np

class SimpleTfidfVectorizer:
    """
    Production-grade, zero-dependency TF-IDF Vectorizer with unigram + bigram support.
    Eliminates C-threadpool/OpenMP issues on bleeding-edge Python versions while delivering
    identical mathematical properties to standard TF-IDF.
    """
    def __init__(self, max_features: int = 5000, min_df: int = 1):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None

    def _tokenize(self, text: str) -> List[str]:
        words = re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())
        tokens = list(words)
        # Add bigrams for contextual phrases (e.g. "battery_health", "apple_id", "no_service")
        for i in range(len(words) - 1):
            tokens.append(f"{words[i]}_{words[i+1]}")
        return tokens

    def fit(self, documents: List[str]):
        doc_tokens = [self._tokenize(doc) for doc in documents]
        df = Counter()
        for tokens in doc_tokens:
            df.update(set(tokens))

        # Filter by min_df and select top max_features
        eligible = [(term, count) for term, count in df.items() if count >= self.min_df]
        eligible.sort(key=lambda x: x[1], reverse=True)
        top_terms = [term for term, _ in eligible[:self.max_features]]

        self.vocab = {term: idx for idx, term in enumerate(top_terms)}
        N = len(documents)
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for term, idx in self.vocab.items():
            self.idf[idx] = np.log((1.0 + N) / (1.0 + df[term])) + 1.0

    def transform(self, documents: List[str]) -> np.ndarray:
        if self.idf is None or not self.vocab:
            raise ValueError("SimpleTfidfVectorizer has not been fitted.")

        mat = np.zeros((len(documents), len(self.vocab)), dtype=np.float32)
        for i, doc in enumerate(documents):
            tokens = self._tokenize(doc)
            tf = Counter(tokens)
            for term, count in tf.items():
                if term in self.vocab:
                    idx = self.vocab[term]
                    mat[i, idx] = count * self.idf[idx]
            # L2 normalization
            norm = np.linalg.norm(mat[i])
            if norm > 0:
                mat[i] /= norm
        return mat

    def fit_transform(self, documents: List[str]) -> np.ndarray:
        self.fit(documents)
        return self.transform(documents)


class SimpleMultinomialNB:
    """
    Self-contained Multinomial Naive Bayes with Laplace smoothing and log-sum-exp stabilization.
    """
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.classes: List[str] = []
        self.class_log_prior: Optional[np.ndarray] = None
        self.feature_log_prob: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: List[str]):
        self.classes = sorted(list(set(y)))
        n_classes = len(self.classes)
        n_features = X.shape[1]

        self.class_log_prior = np.zeros(n_classes, dtype=np.float32)
        self.feature_log_prob = np.zeros((n_classes, n_features), dtype=np.float32)

        for idx, c in enumerate(self.classes):
            mask = np.array([label == c for label in y])
            X_c = X[mask]
            self.class_log_prior[idx] = np.log(len(X_c) / len(y))
            feature_counts = np.sum(X_c, axis=0) + self.alpha
            total_count = np.sum(feature_counts)
            self.feature_log_prob[idx] = np.log(feature_counts / total_count)

    def predict(self, X: np.ndarray) -> List[str]:
        scores = X @ self.feature_log_prob.T + self.class_log_prior
        best_indices = np.argmax(scores, axis=1)
        return [self.classes[i] for i in best_indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        scores = X @ self.feature_log_prob.T + self.class_log_prior
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs


def compute_metrics_helper(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Any]:
    """
    Pure Python/NumPy calculation of accuracy, macro-precision, macro-recall, macro-F1, and weighted-F1.
    """
    total = len(y_true)
    if total == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    acc = correct / total

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        supports.append(support)

    macro_p = float(np.mean(precisions))
    macro_r = float(np.mean(recalls))
    macro_f1 = float(np.mean(f1s))

    total_supp = sum(supports)
    weighted_f1 = float(sum(f1 * s for f1, s in zip(f1s, supports)) / total_supp) if total_supp > 0 else 0.0

    return {
        "accuracy": float(acc),
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class_f1": {label: float(f1) for label, f1 in zip(labels, f1s)}
    }
