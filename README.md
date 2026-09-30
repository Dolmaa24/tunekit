# tunekit

**One-command LoRA/QLoRA fine-tuning for open LLMs.**

```bash
pip install tunekit
tunekit train --model Qwen/Qwen2.5-0.5B-Instruct --data my_data.jsonl
tunekit chat outputs/run
```

That's a fine-tune. tunekit wraps the standard Hugging Face stack (transformers + PEFT + TRL) behind one CLI and one YAML file, so you spend your time on data instead of on trainer boilerplate.

[![CI](https://github.com/Dolmaa24/tunekit/actions/workflows/ci.yml/badge.svg)](https://github.com/Dolmaa24/tunekit/actions)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Dolmaa24/tunekit/blob/main/notebooks/tunekit_colab.ipynb)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

## Why

Fine-tuning a 7B model with QLoRA is ~150 lines of code that everybody rewrites, and most of the bugs live in the parts nobody enjoys: dataset formats, chat templates, pad tokens, dtype/quantisation flags, checkpoint layout. tunekit does those parts once, well:

- **Any common dataset format, auto-detected** — `messages`, alpaca, ShareGPT, prompt/completion, plain text; from `.jsonl`/`.json`/`.csv`/`.parquet`, a directory, or a Hub dataset id. Rows that don't validate are dropped with a reason, not a stack trace.
- **`tunekit validate`** before you burn GPU hours — shows the exact rendered text the model will see, token-length percentiles, and how many rows will be truncated.
- **LoRA or QLoRA** (4-bit / 8-bit) from a single flag; sensible defaults for rank, alpha, scheduler, checkpointing.
- **Reproducible runs** — every output dir gets the resolved config (`tunekit.yaml`), metrics (`tunekit.json`) and a model card.
- **`tunekit eval`** answers "did it help?" — held-out loss and perplexity on the assistant turns, tuned vs base, with sample generations side by side.
- **Ships to where models are used** — `tunekit merge` → `tunekit export --ollama` gives you a GGUF and an Ollama `Modelfile`.
- **Works everywhere the stack works** — CUDA (single or multi-GPU via `accelerate launch`), Apple Silicon (MPS, no quantisation), or CPU for smoke tests.

## Install

```bash
pip install tunekit                # LoRA
pip install "tunekit[quant]"       # + bitsandbytes for QLoRA (Linux + NVIDIA)
pip install "tunekit[wandb]"       # + Weights & Biases logging
```

From source:

```bash
git clone https://github.com/Dolmaa24/tunekit && cd tunekit
pip install -e ".[dev]"
```

## Quickstart

```bash
# 1. see what your hardware supports
tunekit info

# 2. check your data (format detection, token stats, one rendered example)
tunekit validate --model Qwen/Qwen2.5-0.5B-Instruct --data examples/pirate.jsonl

# 3. train (LoRA, bf16, defaults)
tunekit train --model Qwen/Qwen2.5-0.5B-Instruct --data examples/pirate.jsonl --output outputs/pirate

# 4. talk to it
tunekit chat outputs/pirate

# 5. did it help? held-out loss + samples, tuned vs base (writes outputs/pirate/eval.json)
tunekit eval outputs/pirate

# 6. ship it
tunekit merge outputs/pirate outputs/pirate-merged
tunekit export outputs/pirate-merged --quant q8_0 --ollama     # needs a llama.cpp checkout, see below
```

Prefer a config file for anything you'll run twice:

```bash
tunekit init config.yaml        # writes a commented starter
tunekit train config.yaml --set train.lr=1e-4 --set model.load_in_4bit=true
```

Ready-made configs live in [`configs/`](configs/):

| config | model | method | VRAM |
|---|---|---|---|
| `quickstart-smoke.yaml` | SmolLM2-135M | LoRA | any (CPU ok, ~1 min) |
| `qwen2.5-0.5b-lora.yaml` | Qwen2.5-0.5B | LoRA bf16 | ~4 GB |
| `llama-3.2-3b-qlora.yaml` | Llama-3.2-3B | QLoRA 4-bit | ~6 GB (free Colab T4) |
| `qwen2.5-7b-qlora.yaml` | Qwen2.5-7B | QLoRA 4-bit | ~12 GB |
| `hub-dataset-example.yaml` | Qwen2.5-0.5B | LoRA, data from the Hub | ~4 GB |

## Data formats

tunekit reads the first row and picks the converter. All conversational formats become OpenAI-style `messages`; the tokenizer's own chat template is then applied, so the model is trained on exactly the prompt format it will see at inference.

| format | row shape | notes |
|---|---|---|
| `messages` | `{"messages": [{"role": "user", "content": "…"}, {"role": "assistant", "content": "…"}]}` | canonical; multi-turn ok; list-of-parts content is flattened |
| `alpaca` | `{"instruction": "…", "input": "…", "output": "…"}` | `input` optional; also accepts `question`/`answer`, `prompt`/`response`, optional `system` |
| `sharegpt` | `{"conversations": [{"from": "human", "value": "…"}, {"from": "gpt", "value": "…"}]}` | |
| `prompt_completion` | `{"prompt": "…", "completion": "…"}` | loss on completion only; no chat template applied |
| `text` | `{"text": "…"}` | continued pre-training / raw text |

Options in the `data:` section: `system_prompt` (added to every conversation that lacks one), `eval_fraction` / `eval_path`, `max_samples`, `max_length`, and `text_field` / `prompt_field` / `completion_field` for non-standard column names.

## Configuration reference

Everything below is optional except `model.name` and `data.path`. Any key can be overridden with `--set section.key=value`.

```yaml
model:
  name: Qwen/Qwen2.5-0.5B-Instruct
  load_in_4bit: false        # QLoRA (needs tunekit[quant])
  load_in_8bit: false
  dtype: auto                # auto | bfloat16 | float16 | float32
  attn_implementation: null  # flash_attention_2 | sdpa | eager
  trust_remote_code: false
  chat_template: null        # override the tokenizer's Jinja chat template (rarely needed)

data:
  path: data/train.jsonl     # file, directory, or Hub id
  format: auto               # auto | messages | alpaca | sharegpt | prompt_completion | text
  split: train               # Hub datasets only
  subset: null               # Hub dataset config/subset name
  eval_path: null
  eval_fraction: 0.02
  max_samples: null
  system_prompt: null
  max_length: 2048
  shuffle_seed: 42
  text_field: text           # column names, when auto-detection can't find them
  prompt_field: prompt
  completion_field: completion

lora:
  enabled: true              # false = full fine-tune
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear # or [q_proj, k_proj, v_proj, o_proj]
  use_rslora: false
  use_dora: false
  modules_to_save: null      # e.g. [embed_tokens, lm_head]

train:
  output_dir: outputs/run
  epochs: 1
  max_steps: -1              # > 0 overrides epochs
  batch_size: 2
  grad_accum: 8
  lr: 2.0e-4
  scheduler: cosine
  warmup_ratio: 0.03
  weight_decay: 0.0
  max_grad_norm: 1.0
  optimizer: adamw_torch     # paged_adamw_8bit is a good QLoRA choice
  gradient_checkpointing: true
  packing: false
  assistant_only_loss: false # needs a chat template with {% generation %} markers
  logging_steps: 10
  eval_steps: null           # null = every epoch
  save_steps: null
  save_total_limit: 2
  seed: 42
  report_to: [none]          # [wandb] | [tensorboard]
  run_name: null             # defaults to the output_dir's name
  dataloader_num_workers: 0
  resume_from_checkpoint: false
  extra: {}                  # passed straight to TRL's SFTConfig

hub:
  push: false
  repo_id: null
  private: true
```

## Commands

| command | what it does |
|---|---|
| `tunekit info [run_dir] [--model X]` | detected hardware; whether model X's weights fit the GPU; or a finished run's metrics |
| `tunekit init [path]` | write a commented starter config |
| `tunekit validate` | detect/convert the dataset, token-length stats, rendered examples |
| `tunekit train` | fine-tune; `--dry-run` loads data + model and exits |
| `tunekit chat <path>` | interactive chat with an adapter dir, merged dir, or Hub id |
| `tunekit eval <path>` | held-out loss / perplexity + sample generations, tuned vs base; data taken from the run's `tunekit.yaml` unless `--data` is given |
| `tunekit merge <adapter> <out>` | fold the LoRA weights into the base model |
| `tunekit export <merged> [--ollama]` | GGUF via llama.cpp, plus an Ollama Modelfile |
| `tunekit push <dir> <repo_id>` | upload a run directory to the Hugging Face Hub |

`tunekit --help` and `tunekit <command> --help` list every flag.

## Hardware notes

| setup | what works |
|---|---|
| NVIDIA, ≥ 16 GB | everything; 7B QLoRA comfortably, 13B QLoRA with `batch_size: 1` |
| NVIDIA, 8–12 GB (free Colab T4) | LoRA up to ~1.5B in bf16/fp16; QLoRA up to ~7B |
| multiple GPUs | `accelerate launch -m tunekit.cli train config.yaml` (data-parallel) |
| Apple Silicon | LoRA on models that fit in unified memory; **no** 4-bit/8-bit (bitsandbytes is CUDA-only). For serious Mac training use [mlxtuner](https://github.com/Dolmaa24/mlxtuner) |
| CPU | smoke tests only (`configs/quickstart-smoke.yaml`) |

Rules of thumb: QLoRA memory ≈ 0.7 GB per billion params + activations; halve `max_length` or `batch_size` before touching LoRA rank; `lr: 2e-4` for LoRA, `1e-5`–`2e-5` for full fine-tunes.

## GGUF / Ollama export

`tunekit export` shells out to llama.cpp's converter, which is not on PyPI:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp
pip install "tunekit[gguf]"                              # sentencepiece + protobuf, all the converter needs beyond tunekit's stack
tunekit export outputs/merged --quant q8_0 --ollama      # finds ~/llama.cpp automatically
ollama create my-model -f outputs/merged/Modelfile && ollama run my-model
```

Don't install llama.cpp's own `requirements-convert_hf_to_gguf.txt` into the same environment: it pins older `transformers`/`torch` and would downgrade tunekit's. If you prefer their pinned versions, put them in a venv *inside* the checkout (`~/llama.cpp/.venv`) and tunekit will use that interpreter automatically.

For 4-bit GGUFs, run `llama-quantize` from llama.cpp on the q8_0 file. This path was verified end-to-end (train → merge → export → `ollama run`) with SmolLM2-135M.

## Python API

Every command is a thin wrapper over importable functions:

```python
from tunekit.config import RunConfig
from tunekit.train import run
from tunekit.inference import load_for_inference, stream_reply

cfg = RunConfig.from_dict({"model": {"name": "Qwen/Qwen2.5-0.5B-Instruct"}, "data": {"path": "data.jsonl"}})
out_dir = run(cfg)
model, tok = load_for_inference(str(out_dir))
print("".join(stream_reply(model, tok, [{"role": "user", "content": "hello"}])))
```

## FAQ

**Loss goes down but the model doesn't change.** Too few steps or too small a rank for the shift you want; check `tunekit validate` shows the format you expect, then try `epochs: 3`, `lora.r: 32`.

**`CUDA out of memory`.** In order: `load_in_4bit: true` → lower `data.max_length` → `batch_size: 1` with higher `grad_accum` → `optimizer: paged_adamw_8bit`.

**The model rambles / never stops.** The base model's chat template and EOS handling matter; use the `-Instruct` variant of the model, and check your assistant turns don't end with trailing junk.

**How do I know it actually helped?** `tunekit eval outputs/run` scores the held-out split with the tuned and base model and prints both, plus sample generations. Lower loss on assistant turns *and* better-looking samples is the signal; lower loss with worse samples usually means overfitting.

**Where's DPO / RLHF?** Not yet — see [CONTRIBUTING.md](CONTRIBUTING.md) for the roadmap.

## Contributing

Issues and PRs are welcome — new dataset formats, model-family quirks, export targets and hardware reports are all useful. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. Models and datasets you fine-tune keep their own licenses.
