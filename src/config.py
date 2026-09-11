import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORPUS_PATH = PROCESSED_DATA_DIR / "apple_support_corpus.json"
GOLDEN_SET_PATH = DATA_DIR / "golden_eval_set.json"
TAXONOMY_PATH = PROCESSED_DATA_DIR / "intent_taxonomy.json"

# Defined Intent Taxonomy for @AppleSupport
INTENTS = [
    "battery_power",
    "software_update",
    "apple_id_icloud",
    "app_store_billing",
    "connectivity_audio",
    "hardware_device_damage",
    "general_other",
]

INTENT_DESCRIPTIONS = {
    "battery_power": "Battery drain, overheating, charging failure, unexpected shutdown, or battery health degradation.",
    "software_update": "iOS/macOS update stall, post-update bugs, freezing, boot loop, or app crashes after OS update.",
    "apple_id_icloud": "Apple ID lockouts, password reset, 2-factor authentication, iCloud storage sync, or activation lock.",
    "app_store_billing": "Subscription charges, unexpected bill, in-app purchase failure, refund requests, or payment declines.",
    "connectivity_audio": "Wi-Fi disconnections, Bluetooth pairing issues, cellular 'No Service', microphone/speaker distortion.",
    "hardware_device_damage": "Cracked screen, damaged buttons, water ingress, swollen battery, or Genius Bar repair requests.",
    "general_other": "General inquiries, compliments, complaints, store hours, trade-in, or feature feedback.",
}

# Triage & Escalation Enums
ACTION_AUTO_HANDLE = "AUTO_HANDLE"
ACTION_ESCALATE = "ESCALATE"

ESCALATION_REASONS = [
    "requires_dm_auth",          # Needs Apple ID, serial, IMEI, or private credentials
    "hardware_repair_needed",    # Physical damage / defect requiring Genius Bar or depot repair
    "financial_billing_action",  # Refund or subscription dispute needing account billing lookup
    "high_frustration_churn_risk", # Extreme customer distress, abusive rant, or threat to leave
    "complex_multi_tier_issue",  # Persistent failure after troubleshooting steps attempted
]

# Brand Guidelines for @AppleSupport
BRAND_HANDLE = "@AppleSupport"
MAX_TWEET_LENGTH = 280
DM_LINK_TEMPLATE = "https://apple.co/DMSupport"
