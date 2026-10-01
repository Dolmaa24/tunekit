# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [0.1.1] - 2026-10-01

First QLoRA run on a real GPU (Colab T4, Qwen2.5-1.5B-Instruct 4-bit) surfaced these.

### Added
- `chat` and `eval` take `--load-in-4bit`, so a QLoRA adapter runs on the GPU that trained it.
  Without it the base reloads in full precision: ~15 GB for a 7B adapter instead of ~5 GB.
- `peak_vram_gb` recorded in `tunekit.json` and printed when training finishes.
- A clear error when peft refuses to load because the environment has an outdated `torchao`
  (Colab preinstalls one). It only bites at adapter-load time, never during 4-bit training.

### Fixed
- A `data.path` naming a missing local file said the Hub couldn't find it; now it says
  `No such file`.

### Docs
- Colab notebook trains in 4-bit by default (it previously demonstrated plain bf16 LoRA, so the
  QLoRA path it existed to validate was never exercised), no longer blocks on `files.upload()`,
  and installs `tunekit[gguf]` rather than llama.cpp's pins, which downgrade transformers/torch.
- README config reference completed against the schema.

## [0.1.0] - 2026-09-30

First release.

Pre-release polish (30 Sep 2026): verified from a fresh clone — clean install, full test suite, every documented
command, the built wheel outside its source tree, and the whole quickstart end to end. Fixed along the way: a missing
local data file reported a Hub error instead of "no such file"; the README config reference was missing a few real keys.

- `tunekit init / validate / train / eval / chat / merge / export / push / info`.
- YAML run config with `--set section.key=value` overrides (pydantic schema).
- Dataset auto-detection and conversion: `messages`, alpaca, ShareGPT, prompt/completion, text; from `.jsonl`/`.json`/`.csv`/`.parquet`, a directory, or a Hub dataset id.
- LoRA and QLoRA (4-bit / 8-bit via bitsandbytes) through TRL's `SFTTrainer`.
- `eval`: held-out loss / perplexity on the final assistant turn (or completion), sample generations, tuned vs base.
- GGUF export via llama.cpp's converter plus an Ollama Modelfile; verified end-to-end through `ollama run`.
- Colab notebook that trains in 4-bit (QLoRA) by default, evaluates tuned vs base, and exports to GGUF.
- `chat` and `eval` take `--load-in-4bit`, so a QLoRA adapter can be used on the GPU that trained it
  (the base would otherwise load in full precision, ~3x the memory).
- A clear error when peft refuses to load because of an outdated torchao in the environment.
- Example configs, unit + integration tests, CI (push, PR, weekly against latest deps), Dependabot, pre-commit.
