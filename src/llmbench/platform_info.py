"""Platform detection for Mac / Windows / Linux training & inference."""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Literal

OsName = Literal["mac", "windows", "linux", "unknown"]
Accel = Literal["cuda", "mps", "mlx", "cpu"]


@dataclass(frozen=True)
class PlatformInfo:
    os: OsName
    arch: str
    python: str
    accel: Accel
    gpu_name: str | None
    vram_gb: float | None
    ollama: bool
    notes: list[str]

    @property
    def label(self) -> str:
        bits = [self.os, self.arch, self.accel]
        if self.gpu_name:
            bits.append(self.gpu_name)
        if self.vram_gb is not None:
            bits.append(f"{self.vram_gb:.1f}GB")
        return " · ".join(bits)


def detect_os() -> OsName:
    system = platform.system().lower()
    if system == "darwin":
        return "mac"
    if system == "windows":
        return "windows"
    if system == "linux":
        return "linux"
    return "unknown"


def _nvidia_smi() -> tuple[str | None, float | None]:
    if not shutil.which("nvidia-smi"):
        return None, None
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return None, None
    if not out:
        return None, None
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 2:
        return parts[0] if parts else None, None
    try:
        # nvidia-smi memory.total with nounits is MiB
        vram = float(parts[1]) / 1024.0
    except ValueError:
        return parts[0], None
    return parts[0], vram


def _mac_unified_memory_gb() -> float | None:
    if detect_os() != "mac":
        return None
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, timeout=5)
        return int(out.strip()) / (1024**3)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None


def _has_mlx() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def _torch_mps() -> bool:
    try:
        import torch

        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except ImportError:
        return False


def detect_platform() -> PlatformInfo:
    os_name = detect_os()
    arch = platform.machine()
    notes: list[str] = []
    gpu_name: str | None = None
    vram_gb: float | None = None
    accel: Accel = "cpu"

    nv_name, nv_vram = _nvidia_smi()
    if nv_name:
        gpu_name = nv_name
        vram_gb = nv_vram
        accel = "cuda"
        if not _torch_cuda():
            notes.append("NVIDIA GPU found, but PyTorch CUDA is not installed yet.")
    elif os_name == "mac" and arch in {"arm64", "aarch64"}:
        # Prefer the MLX training path on Apple Silicon even before extras are installed.
        gpu_name = "Apple Silicon"
        vram_gb = _mac_unified_memory_gb()
        accel = "mlx"
        if not _has_mlx():
            notes.append(
                "Install MLX extras for Apple Silicon training: pip install 'llmbench[mlx]'"
            )
        elif _torch_mps():
            notes.append("PyTorch MPS is also available for limited workflows.")
    else:
        notes.append(
            "No supported GPU detected. Inference via Ollama may still work; "
            "training will be very slow or blocked."
        )

    if os_name == "windows" and accel == "cuda":
        notes.append("On Windows, training is most reliable inside WSL2.")

    ollama = shutil.which("ollama") is not None
    if not ollama:
        notes.append(
            "Ollama not found on PATH. Install from https://ollama.com for easy local chat."
        )

    return PlatformInfo(
        os=os_name,
        arch=arch,
        python=platform.python_version(),
        accel=accel,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ollama=ollama,
        notes=notes,
    )


def resolve_train_backend(requested: str, info: PlatformInfo) -> str:
    """Map 'auto' to a concrete training backend."""
    if requested != "auto":
        return requested
    if info.accel == "cuda":
        return "transformers"
    if info.accel in {"mlx", "mps"}:
        return "mlx"
    return "transformers"


def resolve_method_for_backend(method: str, backend: str) -> str:
    if backend == "mlx" and method == "qlora":
        return "lora"
    return method
