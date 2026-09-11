import time
from typing import Dict, Any, Optional, NamedTuple, List

from src.intent_classifier import (
    Baseline1MajorityClassifier,
    Baseline2TfidfClassifier,
    ProductionIntentClassifier
)
from src.retriever import GroundedRetriever
from src.triage_engine import TriageEngine, TriageResult, ACTION_AUTO_HANDLE, ACTION_ESCALATE
from src.reply_generator import ReplyGenerator

class AgentResponse(NamedTuple):
    customer_text: str
    predicted_intent: str
    intent_confidence: float
    action: str
    escalation_reason: Optional[str]
    drafted_reply: str
    reply_length: int
    retrieved_evidence: List[Dict[str, Any]]
    processing_time_ms: float

class SupportAgentPipeline:
    """
    Unified Orchestrator for the @AppleSupport AI Agent.
    Supports 3 execution modes:
    - 'baseline1': Trivial Majority Baseline
    - 'baseline2': Simple TF-IDF + 1-NN Baseline
    - 'production': Production Candidate Hybrid System
    """
    def __init__(self, mode: str = "production"):
        self.mode = mode
        self.retriever = GroundedRetriever()
        self.triage_engine = TriageEngine()
        self.reply_generator = ReplyGenerator()

        if mode == "baseline1":
            self.classifier = Baseline1MajorityClassifier()
        elif mode == "baseline2":
            self.classifier = Baseline2TfidfClassifier()
            self.classifier.train()
        else:
            self.classifier = ProductionIntentClassifier()

    def process(self, customer_text: str) -> AgentResponse:
        start_t = time.perf_counter()

        # 1. Intent Classification
        if self.mode == "baseline1":
            intent = self.classifier.predict(customer_text)
            conf = 0.50
        elif self.mode == "baseline2":
            intent = self.classifier.predict(customer_text)
            conf = 0.70
        else:
            intent, conf = self.classifier.predict(customer_text)

        # 2. Historical Context Retrieval
        retrieval_res = self.retriever.extract_actionable_context(customer_text)
        evidence = retrieval_res.get("all_matches", [])

        # 3. Triage & Escalation Decision
        if self.mode == "baseline1":
            # Trivial triage: Always auto-handle
            action = ACTION_AUTO_HANDLE
            reason = None
            triage_res = TriageResult(action=action, reason=reason, confidence=0.5, trigger_rule="trivial_always_autohandle")
        elif self.mode == "baseline2":
            # Simple keyword rule triage
            t_lower = customer_text.lower()
            if any(k in t_lower for k in ["dm", "hacked", "cracked", "shattered", "refund", "stolen"]):
                action = ACTION_ESCALATE
                reason = "requires_dm_auth"
            else:
                action = ACTION_AUTO_HANDLE
                reason = None
            triage_res = TriageResult(action=action, reason=reason, confidence=0.7, trigger_rule="simple_keyword_rule")
        else:
            triage_res = self.triage_engine.evaluate(customer_text, intent)
            action = triage_res.action
            reason = triage_res.reason

        # 4. Reply Drafting
        if self.mode == "baseline1":
            # Trivial generic boilerplate
            reply = "Thanks for reaching out to Apple Support! We would be glad to help. Send us a DM with your device details: https://apple.co/DMSupport."
        elif self.mode == "baseline2":
            # Top-1 nearest neighbor response from historical corpus
            top_reply = retrieval_res.get("primary_resolution")
            reply = top_reply or "We are here for you. Which version of iOS are you running? Check Settings > General > About."
        else:
            reply = self.reply_generator.generate_reply(
                customer_text=customer_text,
                predicted_intent=intent,
                triage_result=triage_res,
                retrieved_context=retrieval_res
            )

        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        return AgentResponse(
            customer_text=customer_text,
            predicted_intent=intent,
            intent_confidence=conf,
            action=action,
            escalation_reason=reason,
            drafted_reply=reply,
            reply_length=len(reply),
            retrieved_evidence=evidence,
            processing_time_ms=elapsed_ms
        )
