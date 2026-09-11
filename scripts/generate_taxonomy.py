import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
from src.config import INTENTS, INTENT_DESCRIPTIONS, TAXONOMY_PATH

taxonomy = {
    "brand": "@AppleSupport",
    "intents": {}
}

examples = {
    "battery_power": [
        "My iPhone X battery drops from 80% to 10% in thirty minutes after the latest update.",
        "Phone gets burning hot while plugged into the official charger and won't charge past 50%.",
        "Battery health says 74% service recommended. Does my phone need a replacement?"
    ],
    "software_update": [
        "Trying to install iOS 11.1 and it has been stuck on the Apple logo for 3 hours.",
        "Ever since updating to the new iOS, all my apps freeze immediately when opened.",
        "Software update failed error occurs every time I try to update over Wi-Fi."
    ],
    "apple_id_icloud": [
        "Locked out of my Apple ID and the two-factor SMS is sending to an old phone number.",
        "How do I reset my iCloud keychain password if I forgot my passcode?",
        "My photos stopped syncing to iCloud storage even though I have 200GB free space."
    ],
    "app_store_billing": [
        "I was charged $14.99 twice for an app subscription I cancelled last week. I want a refund.",
        "My payment method was declined in the App Store but my credit card works fine everywhere else.",
        "How do I get a receipt for an in-app purchase made yesterday?"
    ],
    "connectivity_audio": [
        "AirPods keep disconnecting every 2 minutes from my MacBook Pro during Zoom calls.",
        "My iPhone 8 has said 'No Service' all day even after toggling Airplane mode and restarting.",
        "Speaker crackles loudly whenever someone speaks during a phone call."
    ],
    "hardware_device_damage": [
        "Dropped my iPhone on concrete and the screen is completely shattered, touch doesn't work.",
        "Dropped phone in water, now the screen is flashing green. Can I take it to the Genius Bar?",
        "My iPad home button is physically stuck inside and won't click anymore."
    ],
    "general_other": [
        "What time does the Fifth Avenue Apple Store close on Sundays?",
        "Just wanted to say the camera on the iPhone 11 Pro is absolutely unbelievable!",
        "Do you accept trade-in for an iPhone 7 in store?"
    ]
}

keywords = {
    "battery_power": ["battery", "drain", "charge", "charging", "charger", "overheat", "hot", "percentage", "shut off", "dies", "mah", "health"],
    "software_update": ["update", "ios", "install", "frozen", "freeze", "stuck", "apple logo", "boot", "loop", "restore", "itunes", "version", "upgrade"],
    "apple_id_icloud": ["apple id", "icloud", "password", "passcode", "lock", "locked", "two-factor", "2fa", "verification", "login", "credentials", "account", "keychain"],
    "app_store_billing": ["refund", "charged", "billing", "subscription", "purchase", "receipt", "payment", "card", "declined", "dollar", "renew", "in-app", "money"],
    "connectivity_audio": ["wifi", "wi-fi", "bluetooth", "connect", "disconnect", "cellular", "no service", "signal", "airpods", "speaker", "sound", "volume", "microphone", "mic", "call"],
    "hardware_device_damage": ["shattered", "cracked", "screen", "broken", "drop", "dropped", "water", "damage", "genius bar", "physical", "repair", "button", "bent", "dent"],
    "general_other": ["store", "hours", "open", "trade-in", "shipping", "order", "delivery", "love", "thanks", "complaint", "feedback", "feature"]
}

for intent in INTENTS:
    taxonomy["intents"][intent] = {
        "description": INTENT_DESCRIPTIONS[intent],
        "keywords": keywords[intent],
        "canonical_examples": examples[intent],
        "default_action": "ESCALATE" if intent in ["hardware_device_damage", "app_store_billing"] else "AUTO_HANDLE"
    }

TAXONOMY_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(TAXONOMY_PATH, "w", encoding="utf-8") as f:
    json.dump(taxonomy, f, indent=2)

print("Taxonomy successfully written to", TAXONOMY_PATH)
