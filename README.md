# LLMBench

Cross-platform **local LLM workbench**: pick a model, fine-tune with LoRA/QLoRA, then chat locally.

Same workflow on **macOS (Apple Silicon)**, **Linux + NVIDIA**, and **Windows (WSL2 recommended)**.  
The CLI detects your machine and routes training to the right backend.

```text
config.yaml + models.yaml
        ↓
 llmbench doctor / models / train / chat
        ↓
 ┌──────────────┬────────────────────┐
 │ Mac → MLX    │ NVIDIA → HF+PEFT   │
 │ Infer→Ollama │ Infer→Ollama       │
 └──────────────┴────────────────────┘
```

## Quick start

```bash
# 1) clone & install (core CLI only)
git clone https://github.com/huiwon/LLM.git
cd LLM
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .

# 2) check your machine
llmbench doctor
llmbench models

# 3) create config & try a dry run
llmbench init
llmbench train -c config.yaml --dry-run
```

### Chat (all platforms)

Install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:3b
llmbench chat -c config.yaml
# or: llmbench chat -m qwen2.5-3b
```

### Train

**Linux / Windows (NVIDIA, WSL2 preferred)**

```bash
pip install -e '.[cuda]'
# Install the CUDA build of PyTorch that matches your driver:
# https://pytorch.org/get-started/locally/
llmbench train -c config.yaml
```

**macOS Apple Silicon**

```bash
pip install -e '.[mlx]'
# set backend: mlx (or leave auto) and method: lora in config.yaml
llmbench train -c config.yaml
```

## Config

Copy `config.example.yaml` → `config.yaml` (or run `llmbench init`):

| Field | Meaning |
|-------|---------|
| `model` | Catalog id from `models.yaml` (`qwen2.5-3b`, `llama3.2-3b`, …) |
| `method` | `qlora` (CUDA) or `lora` |
| `backend` | `auto` · `transformers` · `mlx` · `unsloth` |
| `dataset` | JSONL with `instruction` / `input` / `output` |

Sample data: `data/sample_train.jsonl`.

## CLI

| Command | Purpose |
|---------|---------|
| `llmbench doctor` | OS / GPU / package health check |
| `llmbench models` | Models this machine can likely train |
| `llmbench models --all` | Full catalog + blockers |
| `llmbench init` | Write `config.yaml` |
| `llmbench train -c config.yaml` | Fine-tune |
| `llmbench train --dry-run` | Show plan only |
| `llmbench chat` | Ollama streaming chat |

## Platform notes

| Environment | Inference | Training |
|-------------|-----------|----------|
| Linux + NVIDIA | Ollama | `transformers` QLoRA (default) |
| Windows + NVIDIA | Ollama | Same, **WSL2 strongly recommended** |
| Mac Apple Silicon | Ollama / MLX | `mlx` LoRA |
| CPU only | Ollama (slow) | Blocked by `doctor` / `models` |

VRAM numbers in `models.yaml` are **approximate**. `llmbench models` hides combinations that are likely to OOM.

## Project layout

```text
models.yaml              # public model catalog
config.example.yaml      # starter config
data/sample_train.jsonl  # tiny demo dataset
src/llmbench/
  cli.py                 # typer entrypoint
  platform_info.py       # mac / win / linux detection
  catalog.py             # models.yaml loader + filters
  doctor.py
  backends/
    train.py             # transformers / mlx / unsloth adapters
    infer.py             # Ollama chat
```

## Roadmap

- [x] Catalog + platform routing + doctor
- [x] Ollama chat
- [x] CUDA QLoRA via transformers/peft/trl
- [x] MLX LoRA adapter scaffold
- [ ] Merge LoRA adapter into GGUF → Ollama Modelfile
- [ ] Web UI for config / dataset / train progress
- [ ] More trainers (full Unsloth path, LLaMA-Factory)

## License

MIT
