# Hiver AI Customer Support Agent - Complete Setup & Execution Guide

## Overview
This is an end-to-end AI Customer Support Agent for @AppleSupport that classifies customer inquiries, retrieves resolutions, drafts empathetic replies, and makes triage decisions (auto-handle or escalate).

---

## Prerequisites

Before starting, ensure you have:
- **Python 3.10 or higher** (tested on Python 3.10, 3.11, 3.12, 3.13, and 3.14)
- **pip** (Python package manager)
- **Git** (optional, for cloning)
- **Terminal/PowerShell/Command Prompt**

Check your Python version:
```bash
python --version
```

---

## Step 1: Navigate to Project Directory

Open PowerShell or Command Prompt and navigate to your project folder:

```bash
cd "d:\Hiver Assignment1"
```

Verify you're in the correct location by listing files:
```bash
dir
```

You should see: `README.md`, `requirements.txt`, `pyproject.toml`, `src/`, `scripts/`, `data/`, `tests/`

---

## Step 2: Create a Virtual Environment

Creating a virtual environment isolates project dependencies from your system Python:

```bash
python -m venv .venv
```

This creates a `.venv` folder in your project directory.

### Activate the Virtual Environment

**On Windows (PowerShell):**
```bash
.\.venv\Scripts\activate
```

**On Windows (Command Prompt):**
```bash
.venv\Scripts\activate.bat
```

**On Linux/macOS:**
```bash
source .venv/bin/activate
```

**Success indicator:** Your prompt should now show `(.venv)` at the beginning.

---

## Step 3: Install Dependencies

Install all required packages from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `pandas>=2.2.0` — Data manipulation and analysis
- `numpy>=1.26.0` — Numerical computing
- `scikit-learn>=1.4.0` — Machine learning algorithms
- `requests>=2.31.0` — HTTP library for API calls
- `pytest>=8.0.0` — Testing framework

**Installation time:** ~1-2 minutes depending on internet speed.

Verify installation:
```bash
pip list
```

---

## Step 4: Run the Interactive Demo

Experience the AI agent live in your terminal:

```bash
python scripts/interactive_demo.py
```

### Example Interaction

You'll see the prompt:
```
[Customer Tweet] >
```

Type a customer support query (or copy-paste one of these examples):

**Example 1: Battery Issue (AUTO-HANDLE)**
```
My iPhone X battery drops from 100% to 20% in an hour of normal use.
```

Expected output:
```
-----------------------------------------------------------------
Predicted Intent     : battery_power (Confidence: 0.99)
Triage Decision      : AUTO_HANDLE 
Latency              : 2.5ms
Length               : 128/280 characters
[Agent Reply]        : "Check Settings > Battery to see app power usage, and review Settings > Battery > Battery Health. Which iOS version is installed?"
-----------------------------------------------------------------
```

**Example 2: Hardware Damage (ESCALATE)**
```
Dropped my phone on concrete and the screen is completely shattered!
```

Expected output:
```
-----------------------------------------------------------------
Predicted Intent     : hardware_device_damage (Confidence: 0.95)
Triage Decision      : ESCALATE [Reason: hardware_repair_needed]
Latency              : 2.5ms
Length               : 130/280 characters
[Agent Reply]        : "We're here to help explore repair options and schedule a Genius Bar appointment. Please join us in DM: https://apple.co/DMSupport."
-----------------------------------------------------------------
```

**Example 3: PII Leak (ESCALATE with Safety Warning)**
```
My Apple ID is john.doe@sample.com, serial FK2W12345, please reset my password!
```

Expected output:
```
-----------------------------------------------------------------
Predicted Intent     : apple_id_icloud (Confidence: 0.95)
Triage Decision      : ESCALATE [Reason: requires_dm_auth]
Latency              : 2.5ms
Length               : 139/280 characters
[Agent Reply]        : "For your privacy, please remove your personal information from this public tweet. Join us in DM: https://apple.co/DMSupport so we can assist securely."
-----------------------------------------------------------------
```

**To exit the interactive demo:** Press `Ctrl+C` or type `exit`.

---

## Step 5: Run the Evaluation Benchmark

Compare all 3 models (Baseline 1, Baseline 2, and Production Hybrid) on 200 test samples:

```bash
python scripts/run_evaluation.py
```

**Runtime:** ~0.5 seconds (under 1 second total).

### Expected Output

You'll see a comprehensive comparison table:

```
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
```

### What These Metrics Mean

- **Intent Accuracy:** How many customer intents were classified correctly.
- **Macro-F1:** Balance between precision and recall across all 7 intents.
- **Triage Accuracy:** How many auto-handle vs. escalate decisions were correct.
- **Escalation F1:** Quality of escalation decisions (safety matters most).
- **Critical Miss Rate:** False negatives in safety (lower is better).
- **ROUGE-L Score:** Quality of drafted replies compared to reference answers.

---

## Step 6: Run the Test Suite

Verify that all components work correctly:

```bash
pytest tests/ -v
```

This runs all tests with verbose output.

### Run Specific Tests

Test the intent classifiers:
```bash
pytest tests/test_intent.py -v
```

Test the triage/escalation logic:
```bash
pytest tests/test_triage.py -v
```

Test the end-to-end pipeline:
```bash
pytest tests/test_pipeline.py -v
```

Test evaluation metrics:
```bash
pytest tests/test_evaluation.py -v
```

### Expected Result

All tests should pass with `PASSED` status:
```
test_intent.py::test_baseline1 PASSED
test_intent.py::test_baseline2 PASSED
test_intent.py::test_hybrid PASSED
test_triage.py::test_pii_detection PASSED
test_triage.py::test_hardware_escalation PASSED
test_pipeline.py::test_end_to_end PASSED
... (more tests)
```

---

## Step 7: Use the Agent Programmatically (Python Code)

Instead of the CLI, you can use the agent directly in Python scripts:

### Basic Example

Create a file `test_agent.py` in your project root:

```python
from src.pipeline import SupportAgentPipeline

# Initialize the production agent
agent = SupportAgentPipeline(mode="production")

# Process a customer message
response = agent.process("My AirPods keep disconnecting during calls.")

# Display results
print(f"Intent: {response.predicted_intent}")
print(f"Confidence: {response.intent_confidence:.2%}")
print(f"Action: {response.action}")
print(f"Reason: {response.escalation_reason}")
print(f"Reply: {response.drafted_reply}")
print(f"Latency: {response.processing_time_ms:.2f}ms")
```

Run it:
```bash
python test_agent.py
```

Expected output:
```
Intent: connectivity_audio
Confidence: 0.92
Action: AUTO_HANDLE
Reason: None
Reply: Settings > General > Reset Network Settings, then reconnect your AirPods. Does that help?
Latency: 2.45ms
```

### Process Multiple Messages

```python
from src.pipeline import SupportAgentPipeline

agent = SupportAgentPipeline(mode="production")

test_messages = [
    "My iPhone won't update to the latest iOS version.",
    "I was charged twice for my app subscription!",
    "Can't connect to WiFi after the last update.",
]

for msg in test_messages:
    response = agent.process(msg)
    print(f"Tweet: {msg}")
    print(f"Intent: {response.predicted_intent} | Action: {response.action}\n")
```

### Try Different Classifier Modes

```python
# Try Baseline 1 (Majority Class)
agent_b1 = SupportAgentPipeline(mode="baseline1")
response = agent_b1.process("My battery drains too fast")
print(f"Baseline 1: {response.predicted_intent}")

# Try Baseline 2 (TF-IDF + Naive Bayes)
agent_b2 = SupportAgentPipeline(mode="baseline2")
response = agent_b2.process("My battery drains too fast")
print(f"Baseline 2: {response.predicted_intent}")

# Try Production (Hybrid with Regex Rules)
agent_prod = SupportAgentPipeline(mode="production")
response = agent_prod.process("My battery drains too fast")
print(f"Production: {response.predicted_intent}")
```

---

## Step 8: Optional - Download Fresh Data

If you want to re-stream conversation pairs from the source (Hugging Face):

```bash
python scripts/download_data.py
```

**Note:** This is optional. Pre-packaged data is included in `data/processed/`.

---

## Project Structure Quick Reference

```
d:\Hiver Assignment1/
├── data/
│   ├── golden_eval_set.json              # 200 hand-labeled test examples
│   ├── calibration_study_set.json        # 50 calibration samples
│   └── processed/
│       ├── apple_support_corpus.json     # 1,500 historical support pairs
│       └── intent_taxonomy.json          # 7 intent categories
├── src/
│   ├── pipeline.py                       # Main orchestrator
│   ├── intent_classifier.py              # 3 classifier implementations
│   ├── triage_engine.py                  # Escalation logic
│   ├── retriever.py                      # Historical resolution lookup
│   ├── reply_generator.py                # Reply synthesis
│   └── nlp_engine.py                     # Pure NumPy NLP
├── scripts/
│   ├── interactive_demo.py               # Live chat demo
│   ├── run_evaluation.py                 # Benchmark runner
│   └── download_data.py                  # (Optional) Data download
├── tests/
│   ├── test_intent.py                    # Classifier tests
│   ├── test_triage.py                    # Triage logic tests
│   ├── test_pipeline.py                  # End-to-end tests
│   └── test_evaluation.py                # Metric tests
├── README.md                             # Full project documentation
├── REPORT.md                             # Technical report
├── DECISION_LOG.md                       # Engineering decisions
├── requirements.txt                      # Dependencies
└── pyproject.toml                        # Python config
```

---

## 7 Intent Categories

The agent classifies customer messages into these categories:

1. **battery_power** — Battery life, charging, power management issues
2. **software_update** — iOS/macOS updates, upgrade problems
3. **apple_id_icloud** — Account access, authentication, iCloud sync
4. **app_store_billing** — Subscription charges, refunds, payment issues
5. **connectivity_audio** — WiFi, Bluetooth, audio problems
6. **hardware_device_damage** — Physical damage, broken screens, hardware repair
7. **general_other** — All other inquiries

---

## Troubleshooting

### Issue: Python not found
**Solution:** Ensure Python 3.10+ is installed and added to PATH.
```bash
python --version
```

### Issue: Virtual environment won't activate
**Solution:** On Windows PowerShell, you may need to enable script execution:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: Module not found (ImportError)
**Solution:** Ensure virtual environment is activated and dependencies installed:
```bash
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Interactive demo won't start
**Solution:** Make sure all data files exist:
```bash
dir data/processed/
```
You should see `apple_support_corpus.json` and `intent_taxonomy.json`.

### Issue: Tests fail
**Solution:** Run with verbose output to see detailed errors:
```bash
pytest tests/ -v --tb=short
```

---

## Key Metrics Explained

| Metric | What It Measures | Good Value |
|--------|------------------|-----------|
| **Intent Accuracy** | % of correct intent predictions | >0.68 |
| **Triage Accuracy** | % of correct auto-handle vs. escalate decisions | >0.90 |
| **Escalation F1** | Quality of escalation decisions (precision + recall) | >0.75 |
| **Critical Miss Rate** | % of safety-critical issues missed | <0.34 |
| **ROUGE-L** | Similarity of generated replies to reference answers | >0.17 |
| **Processing Latency** | Time to classify + triage + generate reply | <5ms |

---

## Common Tasks Quick Commands

| Task | Command |
|------|---------|
| **Activate virtual env** | `.\.venv\Scripts\activate` |
| **Install dependencies** | `pip install -r requirements.txt` |
| **Run interactive demo** | `python scripts/interactive_demo.py` |
| **Run evaluation** | `python scripts/run_evaluation.py` |
| **Run all tests** | `pytest tests/ -v` |
| **Run intent tests** | `pytest tests/test_intent.py -v` |
| **Run triage tests** | `pytest tests/test_triage.py -v` |
| **Exit virtual env** | `deactivate` |

---

## Next Steps

1. ✅ **Run the interactive demo** to see the agent in action
2. ✅ **Run evaluation** to see benchmark results
3. ✅ **Explore the codebase** — start with `src/pipeline.py` for the main orchestrator
4. ✅ **Read DECISION_LOG.md** for engineering trade-offs
5. ✅ **Read REPORT.md** for the technical deep-dive

---

## Support & Documentation

- **Full project details:** See [README.md](README.md)
- **Technical report:** See [REPORT.md](REPORT.md)
- **Engineering decisions:** See [DECISION_LOG.md](DECISION_LOG.md)
- **Source code:** Browse `src/` directory

---

**Happy testing! 🚀**
