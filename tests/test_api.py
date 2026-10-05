"""Tests for AI Document Chat API."""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import io
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def _register_and_token():
    username = f"user_{os.urandom(4).hex()}"
    r = client.post("/register", json={"username": username, "password": "testpass123"})
    assert r.status_code == 200
    return r.json()["access_token"]


def test_root():
    r = client.get("/")
    assert r.status_code == 200


def test_health():
    r = client.get("/health")
    assert r.status_code == 200


def test_register_and_login():
    token = _register_and_token()
    assert token


def test_upload_and_chat():
    token = _register_and_token()
    headers = {"Authorization": f"Bearer {token}"}

    text = b"The quick brown fox jumps over the lazy dog. RAG is a technique combining retrieval and generation."
    files = {"file": ("test.txt", io.BytesIO(text), "text/plain")}
    r = client.post("/upload", files=files, headers=headers)
    assert r.status_code == 200

    r = client.post("/chat", json={"question": "What is RAG?", "top_k": 3}, headers=headers)
    assert r.status_code == 200
    body = r.json()
    assert "answer" in body
    assert "sources" in body


def test_chat_without_token():
    r = client.post("/chat", json={"question": "What is RAG?"})
    assert r.status_code == 401
