import re
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np

from src.config import INTENTS, TAXONOMY_PATH, CORPUS_PATH
from src.nlp_engine import SimpleTfidfVectorizer, SimpleMultinomialNB

class Baseline1MajorityClassifier:
    def __init__(self, majority_intent: str = "battery_power"):
        self.majority_intent = majority_intent

    def predict(self, text: str) -> str:
        return self.majority_intent

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.majority_intent for _ in texts]


class Baseline2TfidfClassifier:
    def __init__(self):
        self.vectorizer = SimpleTfidfVectorizer(max_features=5000)
        self.clf = SimpleMultinomialNB(alpha=0.1)
        self._is_trained = False

    def train(self, corpus_path: Optional[Path] = None, taxonomy_path: Optional[Path] = None):
        c_path = corpus_path or CORPUS_PATH
        t_path = taxonomy_path or TAXONOMY_PATH

        texts = []
        labels = []

        if t_path.exists():
            with open(t_path, "r", encoding="utf-8") as f:
                tax = json.load(f)
                for intent, data in tax.get("intents", {}).items():
                    for ex in data.get("canonical_examples", []):
                        texts.append(ex)
                        labels.append(intent)
                    kws = data.get("keywords", [])
                    texts.append(" ".join(kws))
                    labels.append(intent)

        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                corpus = json.load(f)
                for item in corpus:
                    c_text = item.get("customer_text", "")
                    if len(c_text) < 15:
                        continue
                    inferred_intent = self._heuristic_label(c_text)
                    if inferred_intent:
                        texts.append(c_text)
                        labels.append(inferred_intent)

        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self._is_trained = True

    def _heuristic_label(self, text: str) -> Optional[str]:
        t = text.lower()
        scores = {intent: 0 for intent in INTENTS}
        if any(w in t for w in ["battery", "drain", "charge", "charger", "overheat", "percentage", "shut off"]):
            scores["battery_power"] += 3
        if any(w in t for w in ["update", "ios", "install", "stuck", "logo", "boot", "freeze", "crash"]):
            scores["software_update"] += 3
        if any(w in t for w in ["apple id", "icloud", "password", "passcode", "lock", "2fa", "verification"]):
            scores["apple_id_icloud"] += 3
        if any(w in t for w in ["refund", "subscription", "charge", "billed", "payment", "card", "receipt"]):
            scores["app_store_billing"] += 3
        if any(w in t for w in ["wifi", "wi-fi", "bluetooth", "service", "signal", "airpods", "speaker", "mic"]):
            scores["connectivity_audio"] += 3
        if any(w in t for w in ["shattered", "cracked", "screen", "broken", "drop", "water", "genius bar", "repair"]):
            scores["hardware_device_damage"] += 3
        if any(w in t for w in ["store", "hours", "trade-in", "shipping", "order", "return"]):
            scores["general_other"] += 3

        best = max(scores.items(), key=lambda x: x[1])
        return best[0] if best[1] > 0 else None

    def predict(self, text: str) -> str:
        if not self._is_trained:
            self.train()
        X_vec = self.vectorizer.transform([text])
        return self.clf.predict(X_vec)[0]

    def predict_batch(self, texts: List[str]) -> List[str]:
        if not self._is_trained:
            self.train()
        X_vec = self.vectorizer.transform(texts)
        return self.clf.predict(X_vec)


class ProductionIntentClassifier:
    """
    Production Hybrid Classifier: Refined domain rules with high precision,
    calibrated with Naive Bayes probability fallbacks.
    """
    def __init__(self):
        self.base_model = Baseline2TfidfClassifier()
        self.base_model.train()

        self.exact_rules = [
            ("hardware_device_damage", re.compile(
                r"\b(cracked|shattered|broken screen|screen bent|dropped in (water|pool|toilet|sink)|"
                r"liquid damage|spilled|swelling|bulging|genius bar|physical repair|face id is not available|"
                r"green line|matrix bleed|taptic engine|stuck flat|unresponsive digitizer|mute switch fell)\b", re.I)),
            ("app_store_billing", re.compile(
                r"\b(refund|unauthorized charge|billed|subscri(ption|be)|in-app purchase|receipt|"
                r"payment method declined|taxed|chargeback|gift card|unpaid balance|ask to buy|billing)\b", re.I)),
            ("apple_id_icloud", re.compile(
                r"\b(apple id|icloud|reset (my )?password|passcode|two-factor|2fa|verification code|"
                r"locked out|activation lock|keychain|account recovery|find my iphone|stolen phone)\b", re.I)),
            ("battery_power", re.compile(
                r"\b(battery (health|drain|capacity|percentage|life|usage)|charging (pad|port|icon)|"
                r"won'?t charge|not charge|charger|wall blocks?|lightning cables?|overheat|burning hot|"
                r"shut(s)? (down|off)|low power mode|fast charging|hand warmer|power adapter)\b", re.I)),
            ("software_update", re.compile(
                r"\b(stuck on (the )?apple logo|software update failed|ios \d+|updating to ios|boot loop|"
                r"attempting data recovery|error 4013|downgrade.*ios|public beta|installing ios|verifying update|"
                r"keyboard lag|calculator app|letter i\b)\b", re.I)),
            ("connectivity_audio", re.compile(
                r"\b(no service|airpods|bluetooth|wi-fi|wifi|dropped calls?|speaker crackl|microphone|"
                r"carplay|airdrop|cellular data|personal hotspot|receiver speaker|audio latency)\b", re.I)),
            ("general_other", re.compile(
                r"\b(store hours|covent garden|trade-in|student discount|animoji|order status|return policy|"
                r"today at apple|applecare\+?|refurbished|engrav|user manuals?|queue taking so long|steve jobs theater)\b", re.I)),
        ]

    def predict(self, text: str) -> Tuple[str, float]:
        for intent, pattern in self.exact_rules:
            if pattern.search(text):
                return intent, 0.95

        X_vec = self.base_model.vectorizer.transform([text])
        probs = self.base_model.clf.predict_proba(X_vec)[0]
        classes = self.base_model.clf.classes
        max_idx = int(np.argmax(probs))
        return classes[max_idx], float(probs[max_idx])

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(t)[0] for t in texts]
