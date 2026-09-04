"""API smoke tests for the Web UI skeleton."""

from fastapi.testclient import TestClient

from llmbench.web.app import create_app


def test_health_and_platform():
    client = TestClient(create_app())
    assert client.get("/api/health").json()["status"] == "ok"
    platform = client.get("/api/platform").json()
    assert "os" in platform
    assert "accel" in platform


def test_models_and_index():
    client = TestClient(create_app())
    models = client.get("/api/models").json()
    assert len(models["models"]) >= 1
    page = client.get("/")
    assert page.status_code == 200
    assert b"LLMBench" in page.content


def test_train_dry_run():
    client = TestClient(create_app())
    res = client.post("/api/train", json={"dry_run": True})
    assert res.status_code == 200
    body = res.json()
    assert body["dry_run"] is True
    assert "plan" in body
