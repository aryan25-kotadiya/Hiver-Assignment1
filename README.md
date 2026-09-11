# Hiver SDE Intern — AI Customer Support Agent

An end-to-end, production-grade AI Customer Support Agent built for **@AppleSupport** using real Twitter customer support conversations from Kaggle (`thoughtvector/customer-support-on-twitter`). The system classifies incoming inquiries into 7 operational domain intents, retrieves historically validated brand resolutions, synthesizes grounded and empathetic public replies (<280 characters), and executes policy-driven triage decisions to safely auto-handle or escalate inquiries with transparent stated reasons.

The repository includes a 200-sample hand-labelled golden evaluation set, an automated evaluation harness comparing the system against two baselines, an LLM-as-a-judge rubric calibrated with human agreement metrics, an interactive CLI demo, and a complete technical report.

---

## 1. Problem

Customer support on public social media presents high operational risk: incoming customer messages are terse, emotionally charged, and publicly visible, while unauthenticated channels cannot securely process sensitive account modifications. 

This project solves the take-home assignment by building an AI support agent for **@AppleSupport** that:
1. **Classifies incoming customer messages** into a defined set of 7 domain intents derived empirically from real Apple customer support threads.
2. **Retrieves historically similar support resolutions** from a curated corpus of real `@AppleSupport` conversations and knowledge base resources.
3. **Drafts grounded, brand-compliant replies** strictly within Twitter's 280-character limit, adopting Apple's calm, courteous, and actionable tone.
4. **Decides whether to AUTO-HANDLE or ESCALATE** each message, recognizing when an issue can be safely guided via public self-service vs. when it requires private human intervention.
5. **Provides an explicit, policy-compliant reason for escalation** (e.g., private credential verification, physical hardware damage, billing disputes, or customer churn threats).

---

## 2. Project Architecture

The pipeline processes incoming customer messages through four decoupled stages:

```text
                     Incoming Customer Tweet
                                │
                                ▼
                   ┌──────────────────────────┐
                   │   Intent Classifier      │
                   │ (Baseline 1, 2, or Hybrid)│
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

### Component Breakdown
1. **Intent Classifier (`src/intent_classifier.py`)**:
   - Classifies customer tweets across 7 operational intents: `battery_power`, `software_update`, `apple_id_icloud`, `app_store_billing`, `connectivity_audio`, `hardware_device_damage`, and `general_other`.
   - Provides 3 swappable implementations: Majority-Class (Baseline 1), TF-IDF + Naive Bayes (Baseline 2), and a Production Hybrid Classifier pairing high-precision regex domain rules with calibrated Naive Bayes probabilities.
2. **Triage & Escalation Engine (`src/triage_engine.py`)**:
   - Decoupled from intent classification to handle cross-cutting policy rules.
   - Identifies sensitive PII leakage, physical hardware damage, billing disputes, high churn risks, and exhausted self-service loops to route to `ESCALATE` with a stated reason (`requires_dm_auth`, `hardware_repair_needed`, `financial_billing_action`, `high_frustration_churn_risk`, `complex_multi_tier_issue`).
3. **Historical Resolution Retriever (`src/retriever.py`)**:
   - Uses TF-IDF vectorization with unigram/bigram tokenization and cosine similarity over 1,500 historical Apple Support pairs to retrieve proven technical resolutions and official Apple Knowledge Base article links.
4. **Reply Generator (`src/reply_generator.py`)**:
   - Formulates responses adhering to brand guidelines (<280 characters, empathetic, never prompts for PII publicly).
   - If escalated, provides safe Direct Message handoff links (`https://apple.co/DMSupport`) and warns customers to remove exposed credentials if PII was detected.
5. **High-Performance NLP Engine (`src/nlp_engine.py`)**:
   - Self-contained, zero-dependency NumPy implementation of TF-IDF vectorization, Multinomial Naive Bayes, and classification metrics. Eliminates C-threadpool/OpenMP deadlocks on Windows and modern Python runtimes (Python 3.10 to 3.14+).

---

## 3. Repository Structure

```text
Hiver Assignment1/
├── data/
│   ├── calibration_study_set.json           # 50 double-blind samples across scores 1-5 for judge calibration
│   ├── evaluation_results.json              # Saved benchmark metrics for all 3 models
│   ├── golden_eval_set.json                 # 200 hand-labelled golden test examples
│   ├── golden_eval_set_sampling_notes.md   # Sampling stratification and annotation methodology
│   └── processed/
│       ├── apple_support_corpus.json        # 1,500 cleaned historical @AppleSupport conversation pairs
│       └── intent_taxonomy.json             # Taxonomy descriptions, keywords, and canonical examples
├── scripts/
│   ├── download_data.py                     # Streaming script to curate pairs from Kaggle/HuggingFace
│   ├── generate_taxonomy.py                 # Generates processed taxonomy configuration
│   ├── interactive_demo.py                  # Terminal interactive CLI to chat with the agent live
│   └── run_evaluation.py                    # Main evaluation benchmark runner (<1s runtime)
├── src/
│   ├── __init__.py
│   ├── config.py                            # Constants, paths, taxonomy definitions, and brand settings
│   ├── data_loader.py                       # Text normalization, HTML unescaping, and tweet pair extractor
│   ├── intent_classifier.py                 # Baseline 1, Baseline 2, and Production Hybrid classifiers
│   ├── nlp_engine.py                        # Pure NumPy TF-IDF, Naive Bayes, and metrics calculator
│   ├── pipeline.py                          # Unified orchestrator (SupportAgentPipeline)
│   ├── reply_generator.py                   # Grounded reply synthesizer and length enforcer
│   ├── retriever.py                         # Grounded historical resolution retriever
│   ├── triage_engine.py                     # Policy triage decider with explicit reason taxonomy
│   └── evaluation/
│       ├── __init__.py
│       ├── agreement_study.py               # Cohen Weighted Kappa, Pearson r, and MAE calculator
│       ├── llm_judge.py                     # 3-dimensional evaluation rubric (Groundedness, Voice, Safety)
│       └── metrics.py                       # Multi-class F1, triage accuracy, ROUGE-L, PII safety rates
├── tests/
│   ├── test_evaluation.py                   # Tests for ROUGE-L, triage metrics, and agreement math
│   ├── test_intent.py                       # Tests for Baseline 1, Baseline 2, and Hybrid classifiers
│   ├── test_pipeline.py                     # Tests for end-to-end pipeline execution across all 3 modes
│   └── test_triage.py                       # Tests for PII, hardware, billing, churn, and auto-handle rules
├── DECISION_LOG.md                          # 12 non-obvious engineering decisions and trade-offs
├── pyproject.toml                           # Package configuration and pytest settings
├── README.md                                # Project overview, quickstart, and reproduction guide
├── REPORT.md                                # 5-section technical report meeting all assignment criteria
└── requirements.txt                         # Runtime dependencies
```

---

## 4. Dataset & Data Preparation

### Data Sources
* **Primary Dataset**: Customer Support on Twitter (`thoughtvector/customer-support-on-twitter` via Kaggle and Hugging Face mirror `SunidhiSriram/twcs`).
* **Selected Brand**: **`@AppleSupport`** (~200k+ customer support interactions).

### Included Data Assets
To ensure headline numbers can be reproduced **offline in under 15 minutes** without downloading multi-gigabyte files or configuring external credentials, the repository comes pre-packaged with:
1. **Curated Historical Support Corpus (`data/processed/apple_support_corpus.json`)**:
   - 1,500 cleaned customer inbound inquiries paired with real `@AppleSupport` resolutions.
   - Anonymized handles removed, HTML entities decoded (`&amp;` -> `&`), and single-turn tweet pairs structured.
2. **Golden Evaluation Set (`data/golden_eval_set.json`)**:
   - Exactly **200 hand-labelled test examples**.
   - 175 examples stratified equally across the 7 intents (25 per intent) + 25 adversarial edge-cases (multi-intent rants, sarcasm, public PII leaks, competitor churn threats, and policy boundaries).
   - Documented in detail in `data/golden_eval_set_sampling_notes.md`.
3. **Calibration Study Set (`data/calibration_study_set.json`)**:
   - 50 double-blind samples with human quality scores ranging across 1.0, 2.0, 3.0, 4.0, and 5.0 to calibrate the automated judge with genuine score variance.

### (Optional) Streaming New Historical Data
If you wish to re-stream or curate fresh conversation pairs directly from the source repository:
```bash
python scripts/download_data.py
```
*(Streams and parses 1,500 clean conversation pairs from Hugging Face in ~10 seconds).*

---

## 5. Installation & Setup

### Prerequisites
* Python 3.10+ (tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14 on Windows, Linux, and macOS).
* Git.

### Setup Instructions
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Hiver-Assignment1.git
cd "Hiver-Assignment1"

# 2. Create and activate a virtual environment
python -m venv .venv

# On Windows (PowerShell):
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 6. How to Run the Pipeline

### Interactive CLI Demo
Experience the live agent directly in your terminal:
```bash
python scripts/interactive_demo.py
```

#### Example Interactive Session:
```text
[Customer Tweet] > My iPhone X battery drops from 100% to 20% in an hour of normal use.
-----------------------------------------------------------------
Predicted Intent     : battery_power (Confidence: 0.99)
Triage Decision      : AUTO_HANDLE 
Latency              : 2.5ms
Length               : 128/280 characters
[Agent Reply]        : "Check Settings > Battery to see app power usage, and review Settings > Battery > Battery Health. Which iOS version is installed?"
-----------------------------------------------------------------

[Customer Tweet] > Dropped my phone on concrete and the screen is completely shattered!
-----------------------------------------------------------------
Predicted Intent     : hardware_device_damage (Confidence: 0.95)
Triage Decision      : ESCALATE [Reason: hardware_repair_needed]
Latency              : 2.5ms
Length               : 130/280 characters
[Agent Reply]        : "We're here to help explore repair options and schedule a Genius Bar appointment. Please join us in DM: https://apple.co/DMSupport."
-----------------------------------------------------------------

[Customer Tweet] > My Apple ID is john.doe@sample.com, serial FK2W12345, please reset my password!
-----------------------------------------------------------------
Predicted Intent     : apple_id_icloud (Confidence: 0.95)
Triage Decision      : ESCALATE [Reason: requires_dm_auth]
Latency              : 2.5ms
Length               : 139/280 characters
[Agent Reply]        : "For your privacy, please remove your personal information from this public tweet. Join us in DM: https://apple.co/DMSupport so we can assist securely."
-----------------------------------------------------------------
```

### Programmatic Python Usage
```python
from src.pipeline import SupportAgentPipeline

# Initialize production hybrid pipeline (or mode='baseline1' / mode='baseline2')
agent = SupportAgentPipeline(mode="production")

response = agent.process("My AirPods keep disconnecting during phone calls.")

print("Intent:", response.predicted_intent)      # connectivity_audio
print("Action:", response.action)                # AUTO_HANDLE
print("Reply:", response.drafted_reply)          # Settings > General > Transfer or Reset iPhone...
print("Latency:", response.processing_time_ms)   # ~2ms
```

---

## 7. Running the Evaluation & Reproducing Headline Results

Run the full evaluation benchmark comparing all 3 models on the 200-sample Golden Evaluation Set:
```bash
python scripts/run_evaluation.py
```
*Execution completes in **under 1 second** (~0.5s total runtime).*

### Actual Benchmark Comparison Output

```text
================================================================================
METRIC                         | BASELINE 1 (Trivial) | BASELINE 2 (Simple) | PROPOSED (Candidate)
--------------------------------------------------------------------------------
Intent Accuracy                | 0.145           | 0.395           | 0.680
Intent Macro-F1                | 0.036           | 0.341           | 0.677
Triage Accuracy                | 0.765           | 0.765           | 0.900
Escalation F1                  | 0.000           | 0.145           | 0.756
Critical Miss Rate (Safety)    | 1.000           | 0.915           | 0.340 (Safer)
False Escalation Rate          | 0.000           | 0.026           | 0.026
ROUGE-L Score                  | 0.136           | 0.104           | 0.179
Length Compliance (<280ch)     | 1.000           | 1.000           | 1.000
PII Safety Rate                | 1.000           | 1.000           | 1.000
LLM-Judge Score (1-5)          | 3.16            | 3.79            | 4.57
Avg Latency per Tweet          | 2.3 ms          | 2.4 ms          | 2.4 ms
Total Benchmark Time           | 0.48 s          | 0.51 s          | 0.51 s
================================================================================
```

### Human vs. Automated Judge Agreement Calibration
Evaluated across 50 double-blind samples with human scores ranging from 1.0 to 5.0:
* **Exact Score Agreement Rate**: **44.0%**
* **Within-1 Point Agreement Rate**: **82.0%**
* **Pearson Correlation ($r$)**: **0.595** (strong positive alignment)
* **Cohen's Linearly Weighted Kappa ($\kappa$)**: **0.312**
* **Mean Absolute Error (MAE)**: **0.85 points**

### Identified Real Failure Modes
The benchmark identifies and logs all edge cases where the proposed candidate deviates from ground truth. Real examples extracted directly from the run:
1. **`GOLD-008` (Symptom vs. Root Cause)**: *"Battery indicator is stuck at 1% even after charging for 4 hours."* (Predicted `software_update` due to 'stuck' correlation vs. True `battery_power`).
2. **`GOLD-013` (Physical Defect Overlap)**: *"The back glass of my iPhone is bulging and pushed outward near the battery!"* (Correctly escalated to DM for hardware repair, but tagged `hardware_device_damage` vs. True `battery_power`).
3. **`GOLD-015` (Dying Device vs. Boot Loop)**: *"My iPhone will not turn on or respond to hard reset after dying completely."* (Predicted `software_update` due to 'hard reset' correlation vs. True `battery_power`).
4. **`GOLD-020` (Replacement Program Pricing)**: *"Can I get a battery replacement for $29 out of warranty?"* (Predicted `AUTO_HANDLE` vs. True `ESCALATE` requiring serial number verification).
5. **`GOLD-023` (Thermal Throttling)**: *"Screen brightness dims automatically because the device is too warm."* (Predicted `software_update` vs. True `battery_power`).

---

## 8. Running Automated Unit Tests

The test suite validates intent classification, triage edge cases, pipeline integration, and evaluation mathematics:
```bash
pytest -v tests/
```

### Test Suite Execution Output:
```text
tests/test_evaluation.py::test_rouge_l_identical PASSED          [  6%]
tests/test_evaluation.py::test_rouge_l_disjoint PASSED           [ 13%]
tests/test_evaluation.py::test_triage_metrics PASSED             [ 20%]
tests/test_evaluation.py::test_agreement_perfect PASSED          [ 26%]
tests/test_intent.py::test_baseline1_majority PASSED             [ 33%]
tests/test_intent.py::test_baseline2_tfidf PASSED                [ 40%]
tests/test_intent.py::test_production_classifier_exact_rules PASSED [ 46%]
tests/test_pipeline.py::test_pipeline_baseline1 PASSED           [ 53%]
tests/test_pipeline.py::test_pipeline_baseline2 PASSED           [ 60%]
tests/test_pipeline.py::test_pipeline_production PASSED          [ 66%]
tests/test_triage.py::test_triage_pii_escalation PASSED          [ 73%]
tests/test_triage.py::test_triage_hardware_damage PASSED         [ 80%]
tests/test_triage.py::test_triage_billing_dispute PASSED         [ 86%]
tests/test_triage.py::test_triage_churn_threat PASSED            [ 93%]
tests/test_triage.py::test_triage_autohandle_safe PASSED         [100%]
====================== 15 passed in 2.32s ======================
```

---

## 9. Deliverables & Documentation Index

In accordance with the assignment instructions, all core deliverables are comprehensively documented in dedicated files:

1. **Comprehensive 5-Section Report ([`REPORT.md`](REPORT.md))**:
   - **Section 1**: Problem framing — what "good" means for `@AppleSupport`, and what we chose NOT to build (no unauthenticated side-effect actions on Twitter).
   - **Section 2**: Results vs. at least two baselines (Trivial Majority and Simple TF-IDF/1-NN).
   - **Section 3**: Failure analysis — top 5 failure modes with real examples and root cause hypotheses.
   - **Section 4**: *"What is misleading about my headline number?"* — a mandatory critical breakdown of class imbalance, ROUGE limitations, judge politeness bias, and multi-turn dialogue degradation.
   - **Section 5**: What we'd do next with one more week (multi-turn session memory, conformal prediction, and human-in-the-loop review queues).
2. **Decision Log ([`DECISION_LOG.md`](DECISION_LOG.md))**:
   - 12 non-obvious engineering and product decisions with detailed technical rationales.
3. **Golden Evaluation Set Notes ([`data/golden_eval_set_sampling_notes.md`](data/golden_eval_set_sampling_notes.md))**:
   - Detailed documentation on the two-tier stratified sampling strategy, annotation guidelines, and class distribution.

---

## 10. Citations & Attribution

* **Customer Support on Twitter Dataset**: Originally curated by `thoughtvector` on Kaggle (`thoughtvector/customer-support-on-twitter`), mirrored on Hugging Face (`SunidhiSriram/twcs`).
* **Banking77 Dataset**: PolyAI (`PolyAI/banking77`) referenced for intent taxonomy design principles.
* **All Pipeline Code**: Authored natively for this assignment with zero third-party boilerplates.