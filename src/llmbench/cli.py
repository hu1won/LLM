"""CLI entrypoint: llmbench doctor | models | train | chat."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from llmbench import __version__
from llmbench.backends.infer import chat_ollama
from llmbench.backends.train import BackendError, run_train
from llmbench.catalog import get_model, load_catalog
from llmbench.config import example_config_path, load_config
from llmbench.doctor import run_doctor
from llmbench.platform_info import (
    detect_platform,
    resolve_method_for_backend,
    resolve_train_backend,
)

app = typer.Typer(
    name="llmbench",
    help="Cross-platform local LLM workbench — pick a model, fine-tune, chat.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.callback()
def _main() -> None:
    """LLMBench CLI."""


@app.command("version")
def version_cmd() -> None:
    """Print package version."""
    console.print(__version__)


@app.command("doctor")
def doctor_cmd() -> None:
    """Detect OS / GPU and show install hints."""
    run_doctor()


@app.command("models")
def models_cmd(
    method: str = typer.Option("qlora", help="lora | qlora"),
    backend: str = typer.Option("auto", help="auto | transformers | mlx | unsloth"),
    all_models: bool = typer.Option(False, "--all", help="Show all catalog entries"),
) -> None:
    """List catalog models, filtered to what this machine can likely train."""
    info = detect_platform()
    resolved = resolve_train_backend(backend, info)
    method_r = resolve_method_for_backend(method, resolved)

    table = Table(title=f"Models ({info.label})", header_style="bold")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Params")
    table.add_column("Train?")
    table.add_column("Reason / notes")

    for m in load_catalog():
        ok, reason = m.is_runnable(info, method_r, resolved)
        if not all_models and not ok:
            continue
        mark = "[green]yes[/green]" if ok else "[red]no[/red]"
        note = reason if not ok else (m.notes or reason)
        table.add_row(m.id, m.name, f"{m.params_b}B", mark, note or "")

    console.print(table)
    console.print(
        f"Resolved train backend: [bold]{resolved}[/bold] · method: [bold]{method_r}[/bold]"
    )
    if not all_models:
        console.print("Tip: [bold]llmbench models --all[/bold] to see blocked models too.")


@app.command("train")
def train_cmd(
    config: Path = typer.Option(
        Path("config.yaml"),
        "--config",
        "-c",
        help="Path to config.yaml (copy from config.example.yaml)",
    ),
    dry_run: bool = typer.Option(False, help="Resolve plan only, do not train"),
) -> None:
    """Fine-tune with LoRA/QLoRA using the best backend for this machine."""
    if not config.exists():
        example = example_config_path()
        console.print(f"[red]Missing {config}[/red]")
        console.print(f"Copy the example first:\n  cp {example} config.yaml")
        raise typer.Exit(code=1)

    cfg = load_config(config)
    info = detect_platform()
    backend = resolve_train_backend(cfg.backend, info)
    method = resolve_method_for_backend(cfg.method, backend)

    try:
        model = get_model(cfg.model)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    ok, reason = model.is_runnable(info, method, backend)
    console.print(f"Platform : {info.label}")
    console.print(f"Model    : {model.id} ({model.hf_id})")
    console.print(f"Backend  : {backend}")
    console.print(f"Method   : {method}")
    console.print(f"Dataset  : {cfg.dataset}")
    console.print(f"Output   : {cfg.output_dir}")
    console.print(f"Runnable : {'yes' if ok else 'no'} ({reason})")

    if dry_run:
        console.print("[yellow]Dry run only — exiting.[/yellow]")
        raise typer.Exit(code=0 if ok else 2)

    if not ok:
        console.print("[red]Refusing to start: environment/model mismatch.[/red]")
        console.print("Run [bold]llmbench doctor[/bold] and [bold]llmbench models --all[/bold].")
        raise typer.Exit(code=2)

    try:
        adapter = run_train(cfg, model, backend, method, info)
    except BackendError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]Done.[/green] Adapter saved to {adapter}")
    if model.ollama_id:
        console.print(
            f"Chat baseline model via Ollama:\n  ollama pull {model.ollama_id}\n  llmbench chat -c {config}"
        )


@app.command("chat")
def chat_cmd(
    config: Path = typer.Option(Path("config.yaml"), "--config", "-c"),
    model_id: Optional[str] = typer.Option(None, "--model", "-m", help="Override catalog id"),
) -> None:
    """Interactive chat via Ollama (base model from catalog)."""
    if config.exists():
        cfg = load_config(config)
        mid = model_id or cfg.model
        settings = cfg.infer
    else:
        if not model_id:
            console.print("Provide -m MODEL_ID or a config.yaml")
            raise typer.Exit(code=1)
        from llmbench.config import BenchConfig

        mid = model_id
        settings = BenchConfig().infer

    try:
        model = get_model(mid)
    except KeyError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Chatting with [bold]{model.name}[/bold] via Ollama ({model.ollama_id}). "
        "Empty line or Ctrl+C to quit."
    )
    messages: list[dict[str, str]] = []
    try:
        while True:
            user = console.input("[bold cyan]you>[/bold cyan] ").strip()
            if not user:
                break
            messages.append({"role": "user", "content": user})
            console.print("[bold green]assistant>[/bold green] ", end="")
            chunks: list[str] = []
            try:
                for piece in chat_ollama(model, settings, messages):
                    console.print(piece, end="")
                    chunks.append(piece)
            except BackendError as exc:
                console.print(f"\n[red]{exc}[/red]")
                raise typer.Exit(code=1) from exc
            console.print()
            messages.append({"role": "assistant", "content": "".join(chunks)})
    except (KeyboardInterrupt, EOFError):
        console.print("\nbye")


@app.command("init")
def init_cmd(
    force: bool = typer.Option(False, help="Overwrite existing config.yaml"),
) -> None:
    """Create config.yaml from the example."""
    target = Path("config.yaml")
    example = example_config_path()
    if target.exists() and not force:
        console.print("config.yaml already exists (use --force to overwrite)")
        raise typer.Exit(code=1)
    target.write_text(example.read_text(encoding="utf-8"), encoding="utf-8")
    console.print(f"[green]Wrote {target}[/green] — edit it, then run llmbench doctor")


@app.command("ui")
def ui_cmd(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(7860, help="Bind port"),
    reload: bool = typer.Option(False, help="Auto-reload (dev)"),
) -> None:
    """Launch the local Web UI (requires: pip install llmbench[web])."""
    try:
        import uvicorn
    except ImportError as exc:
        console.print(
            "[red]Web extras not installed.[/red]\n"
            "  pip install 'llmbench\\[web]'"
        )
        raise typer.Exit(code=1) from exc

    console.print(f"Opening LLMBench UI at [bold]http://{host}:{port}[/bold]")
    uvicorn.run("llmbench.web.app:app", host=host, port=port, reload=reload)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
