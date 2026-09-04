"""Training / inference config schema."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class TrainSettings(BaseModel):
    epochs: int = 1
    learning_rate: float = 2e-4
    batch_size: int = 2
    grad_accum: int = 4
    max_seq_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05


class InferSettings(BaseModel):
    backend: Literal["ollama", "transformers", "mlx"] = "ollama"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.7
    max_tokens: int = 512


class BenchConfig(BaseModel):
    model: str = "qwen2.5-3b"
    method: Literal["lora", "qlora"] = "qlora"
    backend: Literal["auto", "unsloth", "transformers", "mlx"] = "auto"
    dataset: Path = Path("./data/sample_train.jsonl")
    output_dir: Path = Path("./outputs/run-001")
    train: TrainSettings = Field(default_factory=TrainSettings)
    infer: InferSettings = Field(default_factory=InferSettings)


def load_config(path: Path) -> BenchConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return BenchConfig.model_validate(raw)


def example_config_path() -> Path:
    candidates = [
        Path.cwd() / "config.example.yaml",
        Path(__file__).resolve().parents[2] / "config.example.yaml",
    ]
    for path in candidates:
        if path.exists():
            return path
    try:
        data = resources.files("llmbench") / "data" / "config.example.yaml"
        if data.is_file():
            return Path(str(data))
    except (TypeError, FileNotFoundError, ModuleNotFoundError):
        pass
    return candidates[-1]
