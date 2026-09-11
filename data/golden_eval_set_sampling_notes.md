# Golden Evaluation Set — Sampling & Labelling Methodology

## Overview
The golden evaluation set consists of **200 rigorously curated and hand-labelled customer inquiries** directed to `@AppleSupport`, built specifically to benchmark:
1. Multi-class Intent Classification across 7 domain intents.
2. Binary Triage & Escalation Decision (`AUTO_HANDLE` vs. `ESCALATE`).
3. Grounded Reply Quality & Brand Voice Compliance.

---

## 1. Sampling Methodology

The dataset was sampled using a **two-tier stratified strategy**:

### Tier 1: Stratified Intent Sampling (175 Examples, 87.5%)
- **Objective**: Ensure balanced statistical power across the entire operational spectrum of Apple Support.
- **Stratification**: 25 examples allocated to each of the 7 core intents:
  1. `battery_power` (25)
  2. `software_update` (25)
  3. `apple_id_icloud` (25)
  4. `app_store_billing` (25)
  5. `connectivity_audio` (25)
  6. `hardware_device_damage` (25)
  7. `general_other` (25)
- **Source Selection**: Filtered from real customer tweets in Kaggle `twcs.csv` (`thoughtvector/customer-support-on-twitter`). Only conversations with validated `@AppleSupport` replies and clean context were selected.

### Tier 2: Adversarial & Edge-Case Sampling (25 Examples, 12.5%)
- **Objective**: Probe model vulnerabilities under realistic distribution shifts and linguistic noise.
- **5 Failure-Prone Categories (5 samples each)**:
  1. **Multi-Intent Overlaps**: e.g., Post-update battery drain or locked Apple ID with subscription billing disputes.
  2. **Sarcasm & Colloquialisms**: e.g., "Thanks for the free $1200 hand warmer" (overheating).
  3. **Security / PII Traps**: Customers tweeting raw emails, serial numbers, or IMEI codes publicly.
  4. **High-Churn Threats**: Severe customer distress threatening immediate competitor churn ("switching to Android").
  5. **Policy Boundaries**: Questions on jailbroken devices, deceased relative legal inquiries, and unfixable Bluetooth latency in music DAWs.

---

## 2. Labelling Schema & Guidelines

Every record contains:
- `id`: `GOLD-001` through `GOLD-200`.
- `customer_text`: Cleaned customer inquiry text (handles and noisy artifacts removed).
- `true_intent`: One of the 7 mutually exclusive domain intents.
- `true_action`:
  - `AUTO_HANDLE`: Query has a canonical public troubleshooting path, knowledge base solution, or settings adjustment that a customer can execute safely.
  - `ESCALATE`: Query involves sensitive credentials, PII, physical hardware damage, billing transactions, or repeated troubleshooting failure requiring human intervention.
- `true_escalation_reason`:
  - `requires_dm_auth`: Needs Apple ID, serial, IMEI, or private credential verification.
  - `hardware_repair_needed`: Physical damage requiring Genius Bar or mail-in repair.
  - `financial_billing_action`: Subscription refund or billing dispute.
  - `high_frustration_churn_risk`: Urgent churn threat or abusive frustration.
  - `complex_multi_tier_issue`: Persistent failure after repeated self-service steps.
- `reference_reply`: Authoritative ground-truth response adhering to Apple's brand tone (<280 chars, empathetic, actionable).
- `human_quality_score`: Rated on a 1–5 scale (all gold standard references rated 5/5 for calibration).
- `notes`: Specific clinical justification for the ground-truth annotation.

---

## 3. Class Balance Summary
- Total Items: 200
- Intent Distribution:
  - `apple_id_icloud`: 30 (15.0%)
  - `battery_power`: 29 (14.5%)
  - `connectivity_audio`: 29 (14.5%)
  - `hardware_device_damage`: 29 (14.5%)
  - `software_update`: 28 (14.0%)
  - `general_other`: 28 (14.0%)
  - `app_store_billing`: 27 (13.5%)
- Triage Distribution:
  - `AUTO_HANDLE`: 153 (76.5%)
  - `ESCALATE`: 47 (23.5%)
