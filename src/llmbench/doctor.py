"""Environment health checks."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from llmbench.platform_info import PlatformInfo, detect_platform

console = Console()


def _pkg_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def run_doctor() -> PlatformInfo:
    info = detect_platform()

    table = Table(title="LLMBench doctor", show_header=True, header_style="bold")
    table.add_column("Check")
    table.add_column("Status")

    table.add_row("OS", info.os)
    table.add_row("Arch", info.arch)
    table.add_row("Python", info.python)
    table.add_row("Acceleration", info.accel)
    table.add_row("GPU", info.gpu_name or "—")
    table.add_row("Memory / VRAM", f"{info.vram_gb:.1f} GB" if info.vram_gb else "—")
    table.add_row("Ollama CLI", "yes" if info.ollama else "no")

    for pkg in ("torch", "transformers", "peft", "trl", "mlx", "mlx-lm"):
        ver = _pkg_version(pkg)
        table.add_row(f"pip:{pkg}", ver or "not installed")

    console.print(table)

    if info.notes:
        safe = "\n".join(f"• {n.replace('[', '\\[')}" for n in info.notes)
        console.print(Panel(safe, title="Notes", border_style="yellow"))

    if info.accel == "cuda":
        console.print(
            "[green]Suggested extras:[/green] "
            "pip install 'llmbench\\[cuda]' "
            "(use the PyTorch CUDA wheel matching your driver)"
        )
    elif info.os == "mac":
        console.print("[green]Suggested extras:[/green] pip install 'llmbench\\[mlx]'")
    else:
        console.print(
            "[yellow]Training needs a GPU path (CUDA or Apple Silicon). "
            "You can still use `llmbench chat` with Ollama.[/yellow]"
        )

    console.print("\nNext: [bold]llmbench models[/bold]  →  [bold]llmbench train -c config.yaml[/bold]")
    return info
