import os
import re
from typing import Dict, Any, Optional
from src.config import BRAND_HANDLE, MAX_TWEET_LENGTH, DM_LINK_TEMPLATE
from src.triage_engine import TriageResult, ACTION_ESCALATE, ACTION_AUTO_HANDLE

class ReplyGenerator:
    """
    Drafts customer support replies strictly grounded in historical @AppleSupport
    resolutions and brand communication policies (< 280 chars, empathetic, actionable).
    """
    def __init__(self):
        self.dm_link = DM_LINK_TEMPLATE

    def generate_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        triage_result: TriageResult,
        retrieved_context: Optional[Dict[str, Any]] = None
    ) -> str:
        # Check if LLM API is available and enabled
        if os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY"):
            llm_reply = self._generate_with_llm(customer_text, predicted_intent, triage_result, retrieved_context)
            if llm_reply:
                return self._enforce_length_and_format(llm_reply)

        # Production Grounded Rule & Retrieval Engine
        if triage_result.action == ACTION_ESCALATE:
            reply = self._draft_escalation_reply(customer_text, triage_result.reason)
        else:
            reply = self._draft_autohandle_reply(customer_text, predicted_intent, retrieved_context)

        return self._enforce_length_and_format(reply)

    def _draft_escalation_reply(self, customer_text: str, reason: Optional[str]) -> str:
        # Check for public PII in customer text to issue a safety warning
        has_pii = bool(re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|serial", customer_text, re.I))

        if has_pii:
            return f"For your privacy, please remove your personal information from this public tweet. Join us in DM: {self.dm_link} so we can assist securely."

        if reason == "requires_dm_auth":
            return f"We'd be glad to look into your account details safely. Send us a DM here: {self.dm_link} and we'll take a closer look together."

        if reason == "hardware_repair_needed":
            return f"We're here to help explore repair options and schedule a Genius Bar appointment. Please join us in DM: {self.dm_link}."

        if reason == "financial_billing_action":
            return f"We understand your billing concern and want to review this transaction with you. Please DM us: {self.dm_link} so our team can assist."

        if reason == "high_frustration_churn_risk":
            return f"We sincerely apologize for your experience and want to turn this around. Please DM us your case details: {self.dm_link} so an advisor can step in."

        if reason == "complex_multi_tier_issue":
            return f"Since standard steps haven't resolved this, let's take a closer look together in DM: {self.dm_link}."

        return f"We're here to help. Please reach out in DM: {self.dm_link} so we can gather more details safely."

    def _draft_autohandle_reply(
        self,
        customer_text: str,
        predicted_intent: str,
        retrieved_context: Optional[Dict[str, Any]]
    ) -> str:
        text_lower = customer_text.lower()

        # 1. Check for specific highly standard resolutions
        if "iforgot" in text_lower or ("password" in text_lower and "reset" in text_lower):
            return "You can securely reset your password or unlock your account at https://iforgot.apple.com with your trusted number or device."

        if "refund" in text_lower or ("in-app" in text_lower and "accidental" in text_lower):
            return "You can review purchase history and request refunds directly at https://reportaproblem.apple.com by signing in with your Apple ID."

        if "apple logo" in text_lower or "boot loop" in text_lower:
            return "Let's put the device into Recovery Mode and update via iTunes/Finder on a computer without erasing data: support.apple.com/HT201263."

        if "no service" in text_lower:
            return "Toggle Airplane Mode for 10 seconds, restart your phone, and check Settings > General > About for a carrier settings update."

        # 2. Leverage retrieved historical resolution if high confidence
        if retrieved_context and retrieved_context.get("confidence", 0.0) > 0.45:
            cand = retrieved_context.get("primary_resolution", "")
            if cand and len(cand) > 20 and len(cand) <= MAX_TWEET_LENGTH:
                # Clean any lingering user IDs
                cand = re.sub(r"@[A-Za-z0-9_]+", "", cand).strip()
                if not cand.lower().startswith("dm us") and "http" in cand:
                    return cand

        # 3. Intent-Grounded Canonical Best Practice Templates
        intent_responses = {
            "battery_power": "Check Settings > Battery to see app power usage, and review Settings > Battery > Battery Health. Which iOS version is installed?",
            "software_update": "Make sure you have at least 3-5GB free in Settings > General > iPhone Storage, or update via a computer: support.apple.com/HT204204.",
            "apple_id_icloud": "Manage your storage or account security at https://appleid.apple.com or empty your Recently Deleted photos album to free up space.",
            "app_store_billing": "Manage active subscriptions in Settings > [Your Name] > Subscriptions, or view receipts at https://reportaproblem.apple.com.",
            "connectivity_audio": "Try Settings > General > Transfer or Reset iPhone > Reset > Reset Network Settings. Ensure your device is on the latest iOS.",
            "hardware_device_damage": "You can view official screen and hardware repair estimates at support.apple.com/repair and locate authorized providers.",
            "general_other": "We're happy to help! You can find Apple Store hours, manuals, and trade-in values at https://apple.com or let us know device details."
        }

        return intent_responses.get(
            predicted_intent,
            "We're here to help! Could you let us know which Apple device and iOS version you're currently using?"
        )

    def _generate_with_llm(
        self,
        customer_text: str,
        predicted_intent: str,
        triage_result: TriageResult,
        retrieved_context: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        # Placeholder for external LLM API if key is present
        # Gracefully returns None if client library not configured
        return None

    def _enforce_length_and_format(self, text: str) -> str:
        cleaned = re.sub(r"\s+", " ", text).strip()
        if len(cleaned) <= MAX_TWEET_LENGTH:
            return cleaned
        # Truncate gracefully to last sentence or word boundary
        truncated = cleaned[:MAX_TWEET_LENGTH - 3]
        last_punct = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last_punct > 180:
            return truncated[:last_punct + 1]
        return truncated.rstrip() + "..."
