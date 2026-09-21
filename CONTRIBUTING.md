# Contributing to tunekit

Thanks for helping. The goal of this project is *usability*: someone with a dataset and a GPU should get a working fine-tune with one command and no surprises. Keep that bar in mind for every change.

## Setup

```bash
git clone https://github.com/Dolmaa24/tunekit && cd tunekit
python -m venv .venv && source .venv/bin/activate      # or: uv venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Running checks

```bash
ruff check src tests && ruff format src tests     # lint + format
pytest -m "not integration"                       # fast, offline unit tests
pytest -m integration                             # downloads SmolLM2-135M and trains 3 steps (CPU ok, ~1 min)
```

CI runs all three on Linux with CPU torch. Please run them locally before opening a PR.

## Layout

```
src/tunekit/
  config.py     pydantic schema for the YAML config + --set overrides
  data.py       load any file/Hub dataset, detect format, convert to messages/prompt_completion/text
  model.py      hardware detection, tokenizer/model loading, quantisation, LoRA
  train.py      builds TRL SFTConfig/SFTTrainer, saves adapter + metadata + model card
  inference.py  load adapter or merged model, streaming chat
  export.py     merge adapters, GGUF via llama.cpp, Ollama Modelfile
  cli.py        typer commands (thin wrappers over the above)
configs/        example run configs
examples/       tiny sample datasets in each supported format
tests/          unit tests (offline) + one integration test
```

## Good first contributions

- **A new dataset format** — add a detector branch and converter in `data.py`, a row in the README table, and tests. Keep the output one of the three canonical kinds.
- **A model-family quirk** — e.g. a tokenizer that needs a special pad token or template fix. Put it in `model.py` behind a clear condition and add a note in the README hardware/FAQ section.
- **An example config** for a model + GPU combo you've actually run, with the observed VRAM in the header comment.
- **Hardware reports** — open an issue with `tunekit info`, the config, and the step time. These feed the README table.

## Roadmap (help wanted)

- [x] `tunekit eval` — held-out loss / perplexity + sample generations, tuned vs base
- [ ] Preference tuning (DPO/ORPO) via TRL's `DPOTrainer`, reusing the data layer
- [ ] Vision-language models
- [x] `tunekit push` as a standalone command
- [ ] Auto-suggest `batch_size` / `max_length` from detected VRAM (`tunekit info --model` already reports whether the base weights fit; the activation side needs measurements from real GPUs — please open an issue with yours)

## Style

- Python 3.10+, type hints everywhere, `ruff` clean.
- Errors that users can hit should be `DataError` / `RuntimeError` with a message that says what to change, not a traceback.
- Don't add a dependency for something 20 lines of code can do.
- Heavy imports (`trl`, `peft`) stay inside functions so `tunekit --help` is fast.

## Pull requests

Small and focused beats big and sweeping. Describe *what* and *why*; if it changes behaviour, update the README. One approval and green CI is enough to merge.
