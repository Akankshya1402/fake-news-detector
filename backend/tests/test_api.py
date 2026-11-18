"""
API Smoke Tests
===============
Run with:  pytest backend/tests/test_api.py -v
(requires the API server to be running on localhost:8000)
"""

import pytest
import requests

BASE_URL = "http://localhost:8000"


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"


def test_model_info():
    r = requests.get(f"{BASE_URL}/model-info")
    assert r.status_code == 200
    body = r.json()
    assert "model_name" in body
    assert body["num_labels"] == 2
    assert set(body["labels"]) == {"REAL", "FAKE"}


def test_predict_fake():
    payload = {
        "title": "SHOCKING: Aliens Control World Governments!",
        "text": (
            "Anonymous insiders claim that world leaders have been secretly replaced "
            "by extraterrestrial beings. The mainstream media is hiding the truth. "
            "Share this before it gets deleted! Wake up sheeple!"
        ),
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"REAL", "FAKE"}
    assert 0.0 <= body["confidence"] <= 1.0
    assert "REAL" in body["probabilities"]
    assert "FAKE" in body["probabilities"]
    assert "explanation" in body
    assert body["latency_ms"] > 0


def test_predict_real():
    payload = {
        "title": "Federal Reserve raises interest rates by 0.25%",
        "text": (
            "The Federal Reserve announced on Wednesday a 25 basis point increase "
            "in the federal funds rate, citing continued progress toward its 2% "
            "inflation target while acknowledging slowing economic growth."
        ),
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"REAL", "FAKE"}


def test_predict_empty_text():
    r = requests.post(f"{BASE_URL}/predict", json={"text": "   "})
    assert r.status_code == 422


def test_predict_no_title():
    r = requests.post(f"{BASE_URL}/predict", json={"text": "Some news article text here."})
    assert r.status_code == 200
