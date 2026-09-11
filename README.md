# Apple Support AI Customer Support Agent

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 15 Passed](https://img.shields.io/badge/tests-15%20passed-brightgreen.svg)]()
[![Reproducibility: <15s](https://img.shields.io/badge/reproduction-<15s-success.svg)]()

Production-grade, trustworthy AI Customer Support Agent for **@AppleSupport** built on real-world Twitter customer care data from Kaggle (`thoughtvector/customer-support-on-twitter`). 

Developed as part of the **Hiver SDE Intern Take-Home Assignment**.

---

## ⚡ Quickstart: Reproduce Headline Results in < 2 Minutes

The entire evaluation benchmark runs **offline out-of-the-box** without requiring paid API keys, external servers, or large data downloads.

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/<your-username>/hiver-assignment.git
cd "hiver-assignment"

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Benchmark (Reproduce Headline Numbers in ~1 Second)
```bash
python scripts/run_evaluation.py
```

### 3. Run the Automated Test Suite (100% Pass)
```bash
pytest -v tests/
```

### 4. Try the Interactive CLI Agent
```bash
python scripts/interactive_demo.py
```

---

## 📊 Headline Benchmark Results (vs. 2 Baselines)

Evaluated on the **200-sample hand-labelled Golden Evaluation Set** across:
1. **Baseline 1 (Trivial)**: Majority-class intent + Always Auto-Handle + Static boilerplate.
2. **Baseline 2 (Simple)**: TF-IDF + Multinomial Naive Bayes + Keyword triage + 1-NN retrieval.
3. **Proposed System (Candidate)**: Hybrid domain rules & calibrated Naive Bayes + Policy triage engine + Grounded RAG reply drafting.

| Metric | Baseline 1 (Trivial) | Baseline 2 (Simple) | Proposed System (Candidate) |
| :--- | :---: | :---: | :---: |
| **Intent Accuracy** | 14.5% | 39.5% | **68.0%** |
| **Intent Macro-F1** | 0.036 | 0.341 | **0.677** |
| **Triage Accuracy** | 76.5% | 76.5% | **90.0%** |
| **Escalation F1** | 0.000 | 0.145 | **0.756** |
| **Critical Miss Rate (Safety)** | 100.0% | 91.5% | **34.0%** (Safer) |
| **False Escalation Rate** | 0.0% | 2.6% | **2.6%** |
| **ROUGE-L Score** | 0.136 | 0.104 | **0.179** |
| **Length Compliance (<280ch)** | 100.0% | 100.0% | **100.0%** |
| **PII Safety Rate** | 100.0% | 100.0% | **100.0%** |
| **LLM-Judge Score (1–5)** | 3.16 / 5.0 | 3.79 / 5.0 | **4.57 / 5.0** |
| **Latency per Tweet** | 1.3 ms | 1.3 ms | **1.4 ms** |
| **Reproduction Time** | 0.28 s | 0.27 s | **0.30 s** |

### Human vs. LLM-as-a-Judge Calibration (50 Double-Blind Samples)
* **Exact Agreement**: 44.0%
* **Within-1 Point Agreement**: 82.0%
* **Pearson Correlation ($r$)**: **0.595**
* **Cohen's Weighted Kappa**: 0.312
* **Mean Absolute Error (MAE)**: 0.85

---

## 🏛️ System Architecture

```
                     Incoming Customer Tweet
                                │
                                ▼
                   ┌──────────────────────────┐
                   │   Intent Classifier      │
                   │ (Baseline 1, 2, or LLM)  │
                   └────────────┬─────────────┘
                                │ Intent Tag + Confidence
                                ▼
                   ┌──────────────────────────┐
                   │    Triage & Escalation   │
                   │         Engine           │
                   └────────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          ▼                                           ▼
[Action: AUTO_HANDLE]                         [Action: ESCALATE]
          │                                           │
          ▼                                           ▼
┌──────────────────────────┐               ┌──────────────────────────┐
│   Grounded Resolution    │               │ Escalate to Human Agent  │
│      Retriever (RAG)     │               │ (Genius Bar / DM Auth /  │
│ (Historical Apple Data)  │               │ Billing Dispute) + Reason│
└─────────┬────────────────┘               └──────────┬───────────────┘
          │                                           │
          ▼                                           ▼
┌──────────────────────────┐               ┌──────────────────────────┐
│   Draft Grounded Reply   │               │   Draft Empathetic       │
│  (Brand Voice & Rules)   │               │   Handoff Response       │
└──────────────────────────┘               └──────────────────────────┘
```

### 1. Intent Taxonomy (7 Core Operational Classes)
1. `battery_power`: Drain, overheating, charging failure, battery health alerts.
2. `software_update`: Boot loops, update failure, stuck on Apple logo, post-update bugs.
3. `apple_id_icloud`: Password recovery, 2FA codes, account lockout, iCloud storage sync.
4. `app_store_billing`: In-app purchases, duplicate subscriptions, refund disputes.
5. `connectivity_audio`: Wi-Fi drops, Bluetooth pairing, "No Service", AirPods audio.
6. `hardware_device_damage`: Cracked displays, water ingress, bulging batteries, Genius Bar.
7. `general_other`: Store hours, trade-in policy, compliments, general inquiries.

### 2. Triage & Escalation Taxonomy
* `requires_dm_auth`: Customer shares PII (emails, serials) or needs private Apple ID verification.
* `hardware_repair_needed`: Physical damage or component failure requiring Genius Bar service.
* `financial_billing_action`: Refund processing or invoice disputes.
* `high_frustration_churn_risk`: High customer distress or threat to switch to competitors.
* `complex_multi_tier_issue`: Persistent failure after troubleshooting steps attempted.

---

## 📁 Repository Structure

```
.
├── README.md                          # Quickstart & reproduction guide
├── REPORT.md                          # 5-section report (Framing, Baselines, Failures, Caveats)
├── DECISION_LOG.md                    # 12 non-obvious engineering decisions & trade-offs
├── requirements.txt                   # Production dependencies
├── pyproject.toml                     # Pytest configuration & package metadata
├── data/
│   ├── golden_eval_set.json           # 200 hand-labelled golden test examples
│   ├── golden_eval_set_sampling_notes.md # Detailed sampling & annotation documentation
│   ├── calibration_study_set.json     # 50 double-blind samples across scores 1-5
│   ├── evaluation_results.json        # Full benchmark outputs for all 3 models
│   └── processed/
│       ├── apple_support_corpus.json  # 1,500 real AppleSupport conversational pairs
│       └── intent_taxonomy.json       # Formal taxonomy definitions & keywords
├── src/
│   ├── config.py                      # System configuration & thresholds
│   ├── nlp_engine.py                  # Zero-dependency NumPy TF-IDF, Naive Bayes & metrics
│   ├── data_loader.py                 # Streaming/batch processor for Twitter dataset
│   ├── intent_classifier.py           # Baseline 1, Baseline 2, and Production classifiers
│   ├── retriever.py                   # Grounded historical resolution retriever
│   ├── triage_engine.py               # Escalation decider with explicit reason taxonomy
│   ├── reply_generator.py             # Grounded response synthesis (Brand voice & rules)
│   ├── pipeline.py                    # End-to-end support agent orchestrator
│   └── evaluation/
│       ├── metrics.py                 # F1, accuracy, ROUGE-L, PII safety metrics
│       ├── llm_judge.py               # LLM-as-a-judge rubric & scoring harness
│       └── agreement_study.py         # Cohen Kappa, Pearson r & agreement calibration
├── scripts/
│   ├── download_data.py               # Streams & curates AppleSupport pairs from HuggingFace
│   ├── run_evaluation.py              # Main benchmark runner generating headline numbers
│   └── interactive_demo.py            # CLI terminal demo to chat with agent live
└── tests/
    ├── test_pipeline.py               # End-to-end integration tests
    ├── test_intent.py                 # Unit tests for intent classification
    ├── test_triage.py                 # Unit tests for escalation rules
    └── test_evaluation.py             # Unit tests for metrics and scoring
```

---

## 📚 Deliverables Checklist (Assignment Requirements)

- [x] **1. Runnable Pipeline**: Reproduces headline results in under 15 seconds (`python scripts/run_evaluation.py`).
- [x] **2. Golden Evaluation Set**: 200 hand-labelled examples with sampling notes (`data/golden_eval_set.json` & `data/golden_eval_set_sampling_notes.md`).
- [x] **3. Evaluation Harness**: Automated metrics + LLM-as-judge rubric + Human-Judge agreement calibration (`src/evaluation/`).
- [x] **4. Comprehensive Report**: Problem framing, results vs. 2 baselines, top 5 failure modes with real examples, mandatory *"What is misleading about my headline number?"* section, and future roadmap (`REPORT.md`).
- [x] **5. Decision Log**: 12 non-obvious engineering decisions documented with rationale (`DECISION_LOG.md`).

---

## 📜 Citations & Data Attribution
* **Primary Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` via Kaggle & Hugging Face mirror `SunidhiSriram/twcs`).
* **Secondary Dataset Inspiration**: Banking77 (`PolyAI/banking77`) for intent taxonomy stratification.
* **All code authored natively** for this assignment with zero third-party boilerplates.
