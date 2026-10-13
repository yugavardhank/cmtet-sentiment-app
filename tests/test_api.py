"""
test_api.py - Tests for the FastAPI backend using TestClient.
"""
import sys
from pathlib import Path
import pytest

# Allow src imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from fastapi.testclient import TestClient


# ── Fixture ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create test client; skip if model not yet trained."""
    try:
        from main import app
        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Cannot import app (model may not be trained yet): {e}")


# ── Health check ───────────────────────────────────────────────────────────────

def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Valid input ────────────────────────────────────────────────────────────────

def test_valid_input(client):
    resp = client.post("/predict", json={"text": "Movie chala bagundhi bro!"})
    assert resp.status_code == 200
    data = resp.json()
    assert "label" in data
    assert data["label"] in ("NEG", "NTL", "POS")
    assert "probs" in data
    assert "cleaned" in data


def test_valid_long_text(client):
    text = "chala bagundhi " * 20   # 300 chars, within limit
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200


# ── Empty input ────────────────────────────────────────────────────────────────

def test_empty_input(client):
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_whitespace_only(client):
    resp = client.post("/predict", json={"text": "   "})
    assert resp.status_code == 422


# ── Over-length input ──────────────────────────────────────────────────────────

def test_over_length_input(client):
    long_text = "a" * 501
    resp = client.post("/predict", json={"text": long_text})
    assert resp.status_code == 422


def test_exactly_500_chars(client):
    text = "b" * 500
    resp = client.post("/predict", json={"text": text})
    assert resp.status_code == 200


# ── Telugu script input ────────────────────────────────────────────────────────

def test_telugu_script(client):
    resp = client.post("/predict", json={"text": "ఈ సినిమా చాలా బాగుంది!"})
    assert resp.status_code == 200


# ── Emoji only ────────────────────────────────────────────────────────────────

def test_emoji_only(client):
    resp = client.post("/predict", json={"text": "😊🔥❤️"})
    assert resp.status_code == 200


# ── Missing body field ─────────────────────────────────────────────────────────

def test_missing_field(client):
    resp = client.post("/predict", json={})
    assert resp.status_code == 422
