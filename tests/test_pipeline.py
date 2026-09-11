import pytest
from src.pipeline import SupportAgentPipeline

def test_pipeline_baseline1():
    pipe = SupportAgentPipeline(mode="baseline1")
    resp = pipe.process("My iPhone battery is dying")
    assert resp.predicted_intent == "battery_power"
    assert resp.action == "AUTO_HANDLE"
    assert "https://apple.co/DMSupport" in resp.drafted_reply

def test_pipeline_baseline2():
    pipe = SupportAgentPipeline(mode="baseline2")
    resp = pipe.process("My wifi keeps disconnecting")
    assert resp.predicted_intent in ["connectivity_audio", "software_update"]
    assert len(resp.drafted_reply) <= 280

def test_pipeline_production():
    pipe = SupportAgentPipeline(mode="production")
    resp = pipe.process("My iPhone screen is shattered after dropping it")
    assert resp.predicted_intent == "hardware_device_damage"
    assert resp.action == "ESCALATE"
    assert resp.escalation_reason == "hardware_repair_needed"
    assert len(resp.drafted_reply) <= 280
    assert "DM" in resp.drafted_reply
