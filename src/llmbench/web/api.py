"""JSON API for the Web UI."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llmbench import __version__
from llmbench.backends.infer import chat_ollama
from llmbench.backends.train import BackendError, run_train
from llmbench.catalog import get_model
from llmbench.config import BenchConfig, InferSettings
from llmbench.platform_info import (
    detect_platform,
    resolve_method_for_backend,
    resolve_train_backend,
)
from llmbench.service import list_models, platform_dict, read_config, train_plan, write_config

router = APIRouter()


class ConfigUpdate(BaseModel):
    model: str | None = None
    method: Literal["lora", "qlora"] | None = None
    backend: Literal["auto", "unsloth", "transformers", "mlx"] | None = None
    dataset: str | None = None
    output_dir: str | None = None
    train: dict[str, Any] | None = None
    infer: dict[str, Any] | None = None


class TrainRequest(BaseModel):
    dry_run: bool = True
    config: ConfigUpdate | None = None


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    model_id: str | None = None
    messages: list[ChatMessage] = Field(default_factory=list)


def _apply_update(cfg: BenchConfig, update: ConfigUpdate | None) -> BenchConfig:
    if update is None:
        return cfg
    data = cfg.model_dump(mode="python")
    patch = update.model_dump(exclude_none=True)
    if "train" in patch and patch["train"]:
        data["train"] = {**data["train"], **patch.pop("train")}
    if "infer" in patch and patch["infer"]:
        data["infer"] = {**data["infer"], **patch.pop("infer")}
    data.update(patch)
    return BenchConfig.model_validate(data)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/platform")
def platform() -> dict[str, Any]:
    return platform_dict()


@router.get("/models")
def models(
    method: str = "qlora",
    backend: str = "auto",
    show_all: bool = True,
) -> dict[str, Any]:
    return list_models(method=method, backend=backend, show_all=show_all)


@router.get("/config")
def get_config() -> dict[str, Any]:
    path, cfg = read_config()
    return {"path": str(path), "config": cfg.model_dump(mode="json")}


@router.put("/config")
def put_config(update: ConfigUpdate) -> dict[str, Any]:
    path, cfg = read_config()
    cfg = _apply_update(cfg, update)
    path = write_config(cfg, path)
    return {"path": str(path), "config": cfg.model_dump(mode="json")}


@router.post("/train")
def train(req: TrainRequest) -> dict[str, Any]:
    path, cfg = read_config()
    cfg = _apply_update(cfg, req.config)
    plan = train_plan(cfg)

    if req.dry_run:
        return {"dry_run": True, "plan": plan, "message": "Plan only — training not started."}

    if not plan["runnable"]:
        raise HTTPException(status_code=400, detail=plan["reason"])

    info = detect_platform()
    backend = resolve_train_backend(cfg.backend, info)
    method = resolve_method_for_backend(cfg.method, backend)
    model = get_model(cfg.model)
    write_config(cfg, path)

    try:
        adapter = run_train(cfg, model, backend, method, info)
    except BackendError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "dry_run": False,
        "plan": plan,
        "adapter": str(adapter),
        "message": f"Training finished. Adapter at {adapter}",
    }


@router.post("/chat")
def chat(req: ChatRequest) -> StreamingResponse:
    path_cfg = None
    try:
        _, cfg = read_config()
        path_cfg = cfg
    except Exception:
        cfg = BenchConfig()

    model_id = req.model_id or (path_cfg.model if path_cfg else cfg.model)
    settings: InferSettings = path_cfg.infer if path_cfg else cfg.infer

    try:
        model = get_model(model_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    messages = [m.model_dump() for m in req.messages]
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    def event_stream():
        try:
            for chunk in chat_ollama(model, settings, messages):
                yield f"data: {json.dumps({'token': chunk}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except BackendError as exc:
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
