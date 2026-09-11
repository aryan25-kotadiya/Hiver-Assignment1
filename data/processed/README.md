# Processed Apple Support Corpus & Intent Taxonomy

This directory contains the cleaned historical support data and domain taxonomy definitions for **@AppleSupport** used by the AI Customer Support Agent.

---

## 1. Files & Formats

All datasets are provided in multiple formats (**Excel `.xlsx`**, **CSV `.csv`**, and **JSON `.json`**) so users can inspect and edit them easily in spreadsheet software (Microsoft Excel, Google Sheets) or programmatic workflows:

| File | Format | Rows | Description |
| :--- | :---: | :---: | :--- |
| **`apple_support_corpus.xlsx`** | Excel Workbook | 1,500 | Historical 2-turn dialog pairs (`customer_text` + `@AppleSupport` reply) in formatted spreadsheet. |
| **`apple_support_corpus.csv`** | CSV (UTF-8 BOM) | 1,500 | Standard comma-separated version optimized for Excel and data pipelines. |
| **`apple_support_corpus.json`** | JSON | 1,500 | Original JSON array format for direct Python deserialization. |
| **`intent_taxonomy.json`** | JSON | 7 Intents | Formal operational intent taxonomy definitions, keywords, and canonical examples. |

---

## 2. Corpus Schema (`apple_support_corpus.*`)

Each record represents a matched 2-turn customer inquiry and brand resolution:

* `cust_id` *(string)*: Unique identifier of the inbound customer tweet.
* `reply_id` *(string)*: Unique identifier of the corresponding brand reply from `@AppleSupport`.
* `customer_text` *(string)*: Normalized customer inquiry text with handles (`@AppleSupport`) removed, HTML entities decoded (`&amp;` -> `&`), and whitespace normalized.
* `brand_reply` *(string)*: Verified resolution text drafted by an official Apple Support advisor.
* `created_at` *(string)*: Original UTC timestamp of the interaction.

---

## 3. Intent Taxonomy Schema (`intent_taxonomy.json`)

Defines the 7 mutually exclusive operational intents:
1. `battery_power`: Battery drain, overheating, charging failure, battery health degradation.
2. `software_update`: Boot loops, update failures, stuck on Apple logo, post-update bugs.
3. `apple_id_icloud`: Account lockout, 2FA codes, password resets, iCloud sync errors.
4. `app_store_billing`: In-app purchases, duplicate subscriptions, refund disputes.
5. `connectivity_audio`: Wi-Fi disconnections, Bluetooth pairing, "No Service", AirPods audio.
6. `hardware_device_damage`: Cracked glass, water ingress, bulging battery, Genius Bar.
7. `general_other`: Store hours, trade-in value, compliments, general inquiries.

Each intent includes:
- `description`: Formal operational boundary.
- `keywords`: High-frequency n-grams and vocabulary.
- `canonical_examples`: Verified sample customer tweets.
- `default_action`: Default triage tendency (`AUTO_HANDLE` vs. `ESCALATE`).

---

## 4. How to Use & Re-generate

### Reading in Python:
```python
from src.data_loader import load_corpus

# Automatically loads from .xlsx, .csv, or .json
corpus = load_corpus("data/processed/apple_support_corpus.xlsx")
print(f"Loaded {len(corpus)} historical support pairs.")
```

### Re-streaming Fresh Historical Pairs:
```bash
python scripts/download_data.py
python scripts/export_to_sheets.py
```