"""Inference helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator

import httpx

from llmbench.backends.train import BackendError
from llmbench.catalog import ModelEntry
from llmbench.config import InferSettings


def ollama_available(host: str) -> bool:
    try:
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=2.0)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def chat_ollama(
    model: ModelEntry,
    settings: InferSettings,
    messages: list[dict[str, str]],
) -> Iterator[str]:
    if not model.ollama_id:
        raise BackendError(f"Model {model.id} has no ollama_id in the catalog.")
    if not ollama_available(settings.host):
        raise BackendError(
            f"Ollama is not reachable at {settings.host}.\n"
            "Install/start Ollama, then: ollama pull " + (model.ollama_id or "")
        )

    payload = {
        "model": model.ollama_id,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": settings.temperature,
            "num_predict": settings.max_tokens,
        },
    }
    with httpx.stream(
        "POST",
        f"{settings.host.rstrip('/')}/api/chat",
        json=payload,
        timeout=None,
    ) as resp:
        if resp.status_code >= 400:
            body = resp.read().decode("utf-8", errors="replace")
            raise BackendError(f"Ollama error {resp.status_code}: {body}")
        for line in resp.iter_lines():
            if not line:
                continue
            data = json.loads(line)
            msg = data.get("message") or {}
            chunk = msg.get("content") or ""
            if chunk:
                yield chunk
            if data.get("done"):
                break
