import pytest
from src.intent_classifier import (
    Baseline1MajorityClassifier,
    Baseline2TfidfClassifier,
    ProductionIntentClassifier
)

def test_baseline1_majority():
    clf = Baseline1MajorityClassifier()
    assert clf.predict("My screen is cracked") == "battery_power"
    assert len(clf.predict_batch(["test 1", "test 2"])) == 2

def test_baseline2_tfidf():
    clf = Baseline2TfidfClassifier()
    clf.train()
    pred = clf.predict("My battery is dying so quickly after iOS 11")
    assert pred in ["battery_power", "software_update"]

def test_production_classifier_exact_rules():
    clf = ProductionIntentClassifier()
    # Hardware
    pred, conf = clf.predict("The glass screen is completely shattered and cracked")
    assert pred == "hardware_device_damage"
    assert conf >= 0.90

    # Billing
    pred, conf = clf.predict("I want a refund for an unauthorized subscription charge")
    assert pred == "app_store_billing"
    assert conf >= 0.90

    # Apple ID
    pred, conf = clf.predict("I am locked out of my Apple ID and cannot reset password")
    assert pred == "apple_id_icloud"
    assert conf >= 0.90
