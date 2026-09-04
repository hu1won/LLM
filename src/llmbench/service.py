"""Shared service helpers used by CLI and Web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from llmbench.catalog import get_model, load_catalog
from llmbench.config import BenchConfig, example_config_path, load_config
from llmbench.platform_info import (
    PlatformInfo,
    detect_platform,
    resolve_method_for_backend,
    resolve_train_backend,
)


def platform_dict(info: PlatformInfo | None = None) -> dict[str, Any]:
    info = info or detect_platform()
    return {
        "os": info.os,
        "arch": info.arch,
        "python": info.python,
        "accel": info.accel,
        "gpu_name": info.gpu_name,
        "vram_gb": info.vram_gb,
        "ollama": info.ollama,
        "notes": info.notes,
        "label": info.label,
    }


def list_models(
    method: str = "qlora",
    backend: str = "auto",
    show_all: bool = True,
) -> dict[str, Any]:
    info = detect_platform()
    resolved = resolve_train_backend(backend, info)
    method_r = resolve_method_for_backend(method, resolved)
    items = []
    for m in load_catalog():
        ok, reason = m.is_runnable(info, method_r, resolved)
        if not show_all and not ok:
            continue
        items.append(
            {
                "id": m.id,
                "name": m.name,
                "hf_id": m.hf_id,
                "ollama_id": m.ollama_id,
                "params_b": m.params_b,
                "tags": m.tags,
                "notes": m.notes,
                "runnable": ok,
                "reason": reason,
            }
        )
    return {
        "platform": platform_dict(info),
        "backend": resolved,
        "method": method_r,
        "models": items,
    }


def resolve_config_path(path: Path | None = None) -> Path:
    if path is not None:
        return path
    cwd = Path.cwd() / "config.yaml"
    if cwd.exists():
        return cwd
    return Path.cwd() / "config.yaml"


def read_config(path: Path | None = None) -> tuple[Path, BenchConfig]:
    cfg_path = resolve_config_path(path)
    if not cfg_path.exists():
        example = example_config_path()
        cfg_path.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    return cfg_path, load_config(cfg_path)


def write_config(cfg: BenchConfig, path: Path | None = None) -> Path:
    cfg_path = resolve_config_path(path)
    payload = cfg.model_dump(mode="json")
    # Paths as strings for YAML readability
    payload["dataset"] = str(cfg.dataset)
    payload["output_dir"] = str(cfg.output_dir)
    cfg_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return cfg_path


def train_plan(cfg: BenchConfig) -> dict[str, Any]:
    info = detect_platform()
    backend = resolve_train_backend(cfg.backend, info)
    method = resolve_method_for_backend(cfg.method, backend)
    model = get_model(cfg.model)
    ok, reason = model.is_runnable(info, method, backend)
    return {
        "platform": platform_dict(info),
        "model": {
            "id": model.id,
            "name": model.name,
            "hf_id": model.hf_id,
            "ollama_id": model.ollama_id,
        },
        "backend": backend,
        "method": method,
        "dataset": str(cfg.dataset),
        "output_dir": str(cfg.output_dir),
        "runnable": ok,
        "reason": reason,
        "train": cfg.train.model_dump(),
    }
