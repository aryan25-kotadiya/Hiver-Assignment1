import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Any
import numpy as np

from src.config import CORPUS_PATH, TAXONOMY_PATH
from src.nlp_engine import SimpleTfidfVectorizer

class GroundedRetriever:
    """
    Retrieves historically proven resolutions and knowledge base references
    from the curated @AppleSupport corpus using high-speed NumPy cosine similarity.
    """
    def __init__(self, corpus_path: Optional[Path] = None):
        self.corpus_path = corpus_path or CORPUS_PATH
        self.corpus: List[Dict[str, Any]] = []
        self.vectorizer: Optional[SimpleTfidfVectorizer] = None
        self.corpus_vectors: Optional[np.ndarray] = None
        self._is_indexed = False

    def index(self):
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file {self.corpus_path} not found. Run scripts/download_data.py.")

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            self.corpus = json.load(f)

        corpus_texts = []
        for item in self.corpus:
            c_text = item.get("customer_text", "")
            r_text = item.get("brand_reply", "")
            corpus_texts.append(f"{c_text} {r_text}")

        self.vectorizer = SimpleTfidfVectorizer(max_features=6000)
        self.corpus_vectors = self.vectorizer.fit_transform(corpus_texts)
        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._is_indexed:
            self.index()

        query_vec = self.vectorizer.transform([query]) # (1, d)
        # Cosine similarity via dot product since vectors are L2-normalized
        sims = (query_vec @ self.corpus_vectors.T)[0]

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            score = float(sims[idx])
            item = self.corpus[idx]
            results.append({
                "cust_id": item.get("cust_id"),
                "customer_text": item.get("customer_text"),
                "brand_reply": item.get("brand_reply"),
                "score": score
            })
        return results

    def extract_actionable_context(self, query: str) -> Dict[str, Any]:
        top_matches = self.retrieve(query, top_k=3)
        if not top_matches:
            return {"primary_resolution": None, "confidence": 0.0, "sources": []}

        best = top_matches[0]
        return {
            "primary_resolution": best["brand_reply"],
            "confidence": best["score"],
            "all_matches": top_matches
        }
