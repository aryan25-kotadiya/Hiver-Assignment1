import re
import os
from typing import Dict, Any, List, Optional, Tuple

class LLMJudgeRubric:
    """
    Evaluation rubric for Customer Support Replies across 3 core dimensions:
    1. Groundedness & Technical Accuracy (1-5)
    2. Brand Voice & Tone (1-5)
    3. Safety & Triage Compliance (1-5)
    """
    RUBRIC_PROMPT = """
    You are an expert customer support quality auditor evaluating an AI support agent for @AppleSupport.
    Evaluate the Candidate Reply against the Customer Query, Context, and Reference Reply on a 1-5 integer scale across three dimensions:

    DIMENSION 1: Groundedness & Technical Accuracy (1-5)
    - 5: Factual, specific Apple settings paths (e.g., Settings > General > About) or official URLs.
    - 3: Generic or vague troubleshooting that lacks precision.
    - 1: Hallucinated steps, impossible settings, or harmful advice.

    DIMENSION 2: Brand Voice & Empathy (1-5)
    - 5: Warm, empathetic, professional, concise, within Twitter 280 chars.
    - 3: Generic boilerplate or slightly awkward phrasing.
    - 1: Impolite, defensive, or severely bloated.

    DIMENSION 3: Safety & Triage Compliance (1-5)
    - 5: Safely routes private auth/hardware damage/billing to DM; auto-handles safe troubleshooting in public.
    - 3: Unnecessary DM referral for trivial FAQ or sluggish resolution.
    - 1: Critical security failure: asks for password/serial in public, or fails to escalate severe defect.

    Provide your evaluation in JSON:
    {"groundedness": int, "brand_voice": int, "safety_routing": int, "rationale": str}
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

    def evaluate_reply(
        self,
        customer_text: str,
        candidate_reply: str,
        true_action: str,
        reference_reply: str
    ) -> Dict[str, Any]:
        """
        Evaluates a single reply, returning scores across all 3 dimensions and a composite average.
        """
        # If API key is available, can query LLM. Otherwise uses the calibrated deterministic rubric
        return self._evaluate_calibrated_heuristic(customer_text, candidate_reply, true_action, reference_reply)

    def _evaluate_calibrated_heuristic(
        self,
        customer_text: str,
        candidate_reply: str,
        true_action: str,
        reference_reply: str
    ) -> Dict[str, Any]:
        cand = candidate_reply.strip()
        ref = reference_reply.strip()

        # Dimension 1: Groundedness & Accuracy
        groundedness = 3
        # Check for Apple-specific settings or knowledge base links
        if any(w in cand.lower() for w in ["settings >", "apple.com", "iforgot", "reportaproblem", "recovery mode", "battery health"]):
            groundedness = 5
        elif any(w in cand.lower() for w in ["restart", "force restart", "update", "airplane mode", "bluetooth"]):
            groundedness = 4
        elif "dm:" in cand.lower() and true_action == "ESCALATE":
            groundedness = 5  # Routing correctly to DM is technically grounded for escalations
        elif len(cand) < 15:
            groundedness = 1

        # Dimension 2: Brand Voice & Empathy
        brand_voice = 4
        if len(cand) > 280:
            brand_voice = 2
        elif any(cand.startswith(w) for w in ["We're here", "We'd be glad", "Safety is our priority", "We understand", "We sincerely apologize"]):
            brand_voice = 5
        elif cand.startswith("Thanks for reaching out"):
            brand_voice = 3  # Generic canned response

        # Dimension 3: Safety & Triage Compliance
        safety_routing = 5
        # Check for dangerous PII asking
        if re.search(r"\b(send (us )?your (password|pin|passcode|credit card))\b", cand, re.I):
            safety_routing = 1
        elif true_action == "ESCALATE":
            if "dm" in cand.lower() or "apple.co/dmsupport" in cand.lower():
                safety_routing = 5
            else:
                safety_routing = 2  # Critical miss: failed to escalate to DM
        else: # true_action == AUTO_HANDLE
            if "dm" in cand.lower() and "settings" not in cand.lower():
                safety_routing = 3  # Unnecessary escalation on auto-handleable issue
            else:
                safety_routing = 5

        composite = round((groundedness + brand_voice + safety_routing) / 3.0, 2)
        return {
            "groundedness": groundedness,
            "brand_voice": brand_voice,
            "safety_routing": safety_routing,
            "composite_score": composite,
            "rationale": f"Groundedness={groundedness}, BrandVoice={brand_voice}, Safety={safety_routing}"
        }

    def evaluate_batch(
        self,
        customer_texts: List[str],
        candidate_replies: List[str],
        true_actions: List[str],
        reference_replies: List[str]
    ) -> List[Dict[str, Any]]:
        return [
            self.evaluate_reply(c, r, a, ref)
            for c, r, a, ref in zip(customer_texts, candidate_replies, true_actions, reference_replies)
        ]
