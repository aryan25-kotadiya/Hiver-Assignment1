# Decision Log: AI Support Agent for @AppleSupport

This document records the 12 non-obvious architectural, operational, and data decisions made during the design and implementation of the `@AppleSupport` AI customer support agent.

---

### 1. Selection of @AppleSupport as the Focus Brand
* **Decision**: Target `@AppleSupport` exclusively rather than training an ungrounded general-purpose multi-brand agent.
* **Why**: Apple Support is the highest-volume, highest-quality brand in the Kaggle Twitter dataset (~200k+ tweets). It possesses a clean operational separation between public self-service troubleshooting (OS settings, recovery mode) and private authenticated handoffs (serial lookups, billing, Genius Bar repairs).

### 2. Taxonomy Sizing: 7 Mutually Exclusive Domain Intents
* **Decision**: Group inquiries into 7 operational intents (`battery_power`, `software_update`, `apple_id_icloud`, `app_store_billing`, `connectivity_audio`, `hardware_device_damage`, `general_other`) rather than 30+ fine-grained intents (like Banking77).
* **Why**: On Twitter, customer messages are terse (<280 chars) and often combine symptoms. Fine-grained taxonomies suffer from acute label noise (e.g., distinguishing "Wi-Fi drop" from "Network reset"). 7 operational categories map directly to distinct Apple operational routing tiers without semantic overlap.

### 3. Decoupling Intent Classification from Triage & Escalation
* **Decision**: Architect the triage decision engine as an independent downstream layer rather than deriving escalation solely as an intent label (e.g., not having an "escalate" intent).
* **Why**: An issue within `app_store_billing` can be a trivial public FAQ (e.g., "Where is my receipt?" -> `AUTO_HANDLE`) or an urgent financial fraud dispute ("Charged $300 without consent" -> `ESCALATE`). Decoupling allows policy rules and sentiment thresholds to operate across any intent.

### 4. Asymmetric Loss Objective: Minimizing Critical Safety Misses
* **Decision**: Intentionally tune triage thresholds to accept a small false escalation rate (~2.6%) in order to suppress the Critical Miss Rate (predicting `AUTO_HANDLE` on an issue that requires `ESCALATE`).
* **Why**: In real-world customer support, an unnecessary DM escalation costs ~$3 in human agent time, whereas auto-handling a compromised Apple ID or a swollen lithium battery risks account theft, customer churn, or physical safety hazards.

### 5. Pure NumPy NLP Engine over Heavy C-Extensions
* **Decision**: Build a self-contained, high-performance TF-IDF vectorizer and Naive Bayes classifier in pure NumPy (`src/nlp_engine.py`) rather than relying on external scikit-learn C-binaries.
* **Why**: Prevents OpenMP/joblib threadpool deadlocks on Windows and newer Python runtimes (Python 3.14+). It cuts cold-start pipeline import time from >20s to <50ms and guarantees evaluators can run the benchmark in <1 second without binary compilation errors.

### 6. Proactive Public PII Detection & Deletion Warning
* **Decision**: If a customer tweets an email address, password, or Apple hardware serial number publicly, the agent immediately instructs them to delete the public tweet in addition to escalating to DM.
* **Why**: Standard deflection to DM leaves the customer's exposed credentials visible on Twitter's public timeline. A trustworthy support agent must prioritize customer security over conversational brevity.

### 7. Explicit Decision NOT to Build Autonomous Action Execution
* **Decision**: Reject autonomous execution of account actions (e.g., issuing refunds, unlocking iCloud accounts, booking Genius Bar appointments) via Twitter bot.
* **Why**: Twitter accounts are not authenticated identities. Fulfilling transactional requests from an unauthenticated tweet without 2-factor authentication creates an immediate account takeover vulnerability. Twitter is an inbound routing and triage channel, not an execution backend.

### 8. Historical Resolution RAG over Unconstrained Generation
* **Decision**: Condition reply generation on historical `@AppleSupport` resolutions and official knowledge base articles (`support.apple.com/HT...`) rather than unconstrained open-ended LLM completion.
* **Why**: Unconstrained LLMs frequently hallucinate settings menus (e.g., invent non-existent iOS toggles) or suggest hazardous fixes. Grounding in historical brand responses ensures the agent suggests actual Apple repair flows and official URLs.

### 9. Multi-Dimensional Rubric for LLM-as-a-Judge
* **Decision**: Decompose reply quality evaluation into three discrete dimensions (Groundedness/Accuracy, Brand Voice, Safety/Triage) rather than a single monolithic "quality" score.
* **Why**: Monolithic scores conflate tone with correctness. An LLM might give 5/5 to a polite, beautifully written response that provides completely hallucinated technical advice. Separating dimensions surfaces critical technical and safety defects.

### 10. Varied Score Spectrum for Human-Judge Agreement Calibration
* **Decision**: Construct a 50-sample calibration set with a deliberate, documented score distribution across 1.0, 2.0, 3.0, 4.0, and 5.0 ratings.
* **Why**: If human evaluators only score gold-standard responses (all 5.0), the human score variance is zero, causing Pearson correlation ($r$) to be mathematically undefined ($0/0$). Measuring agreement across poor, mediocre, and excellent replies validates the judge's true discriminative power ($r = 0.595$, Within-1 Agreement = 82.0%).

### 11. Two-Tier Stratified Golden Set Sampling Strategy
* **Decision**: Build the 200-sample golden evaluation set with 175 stratified examples (25 per intent) and 25 adversarial edge cases (sarcasm, PII traps, multi-intent rants, competitor threats).
* **Why**: Random sampling from Twitter over-indexes on high-frequency boilerplate ("Thanks for reaching out") and misses low-frequency, high-consequence failure modes like battery thermal venting or law enforcement subpoena requests.

### 12. Zero-Cost, Instant (<15s) Evaluation Guarantee
* **Decision**: Package the preprocessed corpus and golden set directly in the repository with a high-speed local evaluation harness that runs without requiring commercial LLM API keys.
* **Why**: Evaluators assessing take-home assignments should not be forced to provide private credit card API keys, configure Ollama, or wait 30 minutes for slow network downloads. The headline numbers must be fully verifiable in seconds.
