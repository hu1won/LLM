"""Model catalog loaded from models.yaml."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from llmbench.platform_info import PlatformInfo


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_catalog_path() -> Path:
    candidates = [
        Path.cwd() / "models.yaml",
        _repo_root() / "models.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    # Installed wheel force-include
    try:
        data = resources.files("llmbench") / "data" / "models.yaml"
        if data.is_file():
            return Path(str(data))
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    return _repo_root() / "models.yaml"


class ModelEntry(BaseModel):
    id: str
    name: str
    hf_id: str
    ollama_id: str | None = None
    params_b: float
    min_vram_gb: dict[str, float] = Field(default_factory=dict)
    train_backends: list[str] = Field(default_factory=list)
    infer_backends: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None

    def min_vram_for(self, method: str, backend: str) -> float | None:
        if backend == "mlx":
            return self.min_vram_gb.get("mlx")
        return self.min_vram_gb.get(method)

    def is_runnable(self, info: PlatformInfo, method: str, backend: str) -> tuple[bool, str]:
        if backend not in self.train_backends and backend != "auto":
            return False, f"backend '{backend}' not listed for this model"

        if info.accel == "cpu":
            return False, "no GPU / Apple Silicon acceleration detected"

        if info.accel in {"mlx", "mps"} and backend not in {"mlx", "auto"}:
            return False, "on Apple Silicon prefer backend=mlx (or auto)"

        if info.accel == "cuda" and backend == "mlx":
            return False, "mlx is for Apple Silicon only"

        if info.os == "mac" and backend == "transformers":
            return False, "on Mac use backend=mlx (transformers path is CUDA-oriented)"

        key_method = "lora" if backend == "mlx" else method
        needed = self.min_vram_for(key_method, backend)
        if needed is not None and info.vram_gb is not None and info.vram_gb + 0.5 < needed:
            return False, f"needs ~{needed:.0f}GB, machine has ~{info.vram_gb:.1f}GB"

        return True, "ok"


def load_catalog(path: Path | None = None) -> list[ModelEntry]:
    catalog_path = path or default_catalog_path()
    if not catalog_path.exists():
        raise FileNotFoundError(f"models.yaml not found at {catalog_path}")
    raw: dict[str, Any] = yaml.safe_load(catalog_path.read_text(encoding="utf-8")) or {}
    return [ModelEntry.model_validate(item) for item in raw.get("models", [])]


def get_model(model_id: str, path: Path | None = None) -> ModelEntry:
    for entry in load_catalog(path):
        if entry.id == model_id:
            return entry
    known = ", ".join(m.id for m in load_catalog(path))
    raise KeyError(f"Unknown model id '{model_id}'. Known: {known}")
