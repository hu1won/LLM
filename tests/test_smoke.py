"""Smoke tests that need no GPU."""

from llmbench.catalog import load_catalog
from llmbench.config import BenchConfig, load_config
from llmbench.platform_info import detect_platform, resolve_train_backend


def test_catalog_loads():
    models = load_catalog()
    assert len(models) >= 3
    assert any(m.id == "qwen2.5-3b" for m in models)


def test_example_config(tmp_path):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "config.example.yaml")
    assert cfg.model == "qwen2.5-3b"
    assert isinstance(cfg, BenchConfig)


def test_platform_resolve():
    info = detect_platform()
    backend = resolve_train_backend("auto", info)
    assert backend in {"transformers", "mlx", "unsloth"}
