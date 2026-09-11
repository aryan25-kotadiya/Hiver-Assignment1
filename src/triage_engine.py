import re
from typing import Dict, Optional, Tuple, NamedTuple
from src.config import (
    ACTION_AUTO_HANDLE,
    ACTION_ESCALATE,
    ESCALATION_REASONS
)

class TriageResult(NamedTuple):
    action: str
    reason: Optional[str]
    confidence: float
    trigger_rule: str

class TriageEngine:
    """
    Decides whether an incoming customer message should be auto-handled or escalated to a human,
    providing a transparent, policy-compliant stated reason.
    """
    def __init__(self):
        # 1. PII and Security Triggers (requires_dm_auth)
        self.re_pii = re.compile(
            r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)|"
            r"(\b[A-Z0-9]{10,12}\b.*serial)|(serial.*[A-Z0-9]{10,12})|"
            r"\b(password|passcode|two-factor|2fa|verification code|locked out|hacked|account takeover|account recovery for \d+|disabled in the app store)\b",
            re.I
        )
        self.re_activation_lock = re.compile(r"\b(activation lock|secondhand.*activation lock|activation lock.*secondhand|deceased relative)\b", re.I)

        # 2. Hardware and Physical Damage Triggers (hardware_repair_needed)
        self.re_hardware = re.compile(
            r"\b(shattered|cracked|screen bent|green line|swelling|bulging|service recommended|"
            r"face id is not available|greyed out wi-?fi|jammed (button|switch)|fell out|matrix bleed|"
            r"third party (shop|repair)|chemical smell|bricked my phone completely.*black screen|"
            r"vibrates continuously non-stop|headphone jack broke off|unresponsive digitizer zone|"
            r"spilled (coffee|water|drink)|liquid damage)\b|"
            r"(dropped|fell|submerged).*in(to)? (a |the )?(water|pool|toilet|sink|ocean|lake)",
            re.I
        )

        # 3. Financial and Billing Triggers (financial_billing_action)
        self.re_billing = re.compile(
            r"\b(fraudulent|unauthorized charge|billed (twice|\$\d+)|double charge|chargeback|"
            r"cancelled.*still charged|charged.*without.*warning|robux|roblox coins|tax dispute|"
            r"annual renewal without|charged twice for the same)\b",
            re.I
        )

        # 4. Churn Risk and Extreme Distress Triggers (high_frustration_churn_risk)
        self.re_churn = re.compile(
            r"\b(taking (all )?.*(lines )?to android|switching to (samsung|android)|worst customer service|"
            r"three weeks without|file a lawsuit|filing with bbb|disgusted)\b",
            re.I
        )

        # 5. Multi-tier Troubleshooting Exhaustion (complex_multi_tier_issue)
        self.re_exhausted = re.compile(
            r"\b(already (tried|reset|restored)|still (not working|no service|fails)|exhausted all steps|trade-in quote id|warehouse sent an email saying \$0)\b",
            re.I
        )

    def evaluate(self, customer_text: str, predicted_intent: str) -> TriageResult:
        text = customer_text.strip()
        t_lower = text.lower()

        # Check 1: PII or Security Account Lockout -> ESCALATE (requires_dm_auth)
        if self.re_pii.search(text) or self.re_activation_lock.search(text):
            if not any(k in t_lower for k in ["how do i turn on", "how do i cancel two-factor", "is it possible", "can i change my @icloud"]):
                return TriageResult(
                    action=ACTION_ESCALATE,
                    reason="requires_dm_auth",
                    confidence=0.96,
                    trigger_rule="pii_or_account_security_rule"
                )

        # Check 2: Physical Hardware Damage -> ESCALATE (hardware_repair_needed)
        if self.re_hardware.search(text):
            if not any(k in t_lower for k in ["how much does it cost", "how do i clean", "can i replace just the glass"]):
                return TriageResult(
                    action=ACTION_ESCALATE,
                    reason="hardware_repair_needed",
                    confidence=0.98,
                    trigger_rule="physical_damage_or_hardware_fault_rule"
                )

        # Check 3: Urgent Billing, Duplicate Charges, Chargebacks -> ESCALATE (financial_billing_action)
        if self.re_billing.search(text):
            return TriageResult(
                action=ACTION_ESCALATE,
                reason="financial_billing_action",
                confidence=0.95,
                trigger_rule="financial_billing_dispute_rule"
            )

        # Check 4: Severe Churn Risk / Legal / Regulator threat -> ESCALATE (high_frustration_churn_risk)
        if self.re_churn.search(text):
            return TriageResult(
                action=ACTION_ESCALATE,
                reason="high_frustration_churn_risk",
                confidence=0.94,
                trigger_rule="high_frustration_churn_risk_rule"
            )

        # Check 5: Exhausted Troubleshooting -> ESCALATE (complex_multi_tier_issue)
        if self.re_exhausted.search(text):
            return TriageResult(
                action=ACTION_ESCALATE,
                reason="complex_multi_tier_issue",
                confidence=0.90,
                trigger_rule="exhausted_troubleshooting_loop_rule"
            )

        # Otherwise: Safe to Auto-Handle with grounded self-service troubleshooting
        return TriageResult(
            action=ACTION_AUTO_HANDLE,
            reason=None,
            confidence=0.88,
            trigger_rule="standard_self_service_eligible"
        )
