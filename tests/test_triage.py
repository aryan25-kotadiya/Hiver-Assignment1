import pytest
from src.triage_engine import TriageEngine, ACTION_ESCALATE, ACTION_AUTO_HANDLE

def test_triage_pii_escalation():
    engine = TriageEngine()
    res = engine.evaluate("My email is user@domain.com and serial is FK2W12345678, help me login!", "apple_id_icloud")
    assert res.action == ACTION_ESCALATE
    assert res.reason == "requires_dm_auth"

def test_triage_hardware_damage():
    engine = TriageEngine()
    res = engine.evaluate("Dropped my iPhone into the pool and the screen is flickering", "hardware_device_damage")
    assert res.action == ACTION_ESCALATE
    assert res.reason == "hardware_repair_needed"

def test_triage_billing_dispute():
    engine = TriageEngine()
    res = engine.evaluate("Unauthorized fraudulent charge of $150 on my credit card from iTunes", "app_store_billing")
    assert res.action == ACTION_ESCALATE
    assert res.reason == "financial_billing_action"

def test_triage_churn_threat():
    engine = TriageEngine()
    res = engine.evaluate("Worst customer service ever! Taking all my lines to Android tomorrow!", "general_other")
    assert res.action == ACTION_ESCALATE
    assert res.reason == "high_frustration_churn_risk"

def test_triage_autohandle_safe():
    engine = TriageEngine()
    res = engine.evaluate("What time does the Apple store close in Chicago?", "general_other")
    assert res.action == ACTION_AUTO_HANDLE
    assert res.reason is None
