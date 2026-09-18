"""Typed configuration for a fine-tuning run.

A run is described by a single YAML file (see ``configs/``). Every field has a
sensible default so a minimal config only needs ``model.name`` and ``data.path``.
Any field can be overridden from the CLI with ``--set section.key=value``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DataFormat = Literal["auto", "messages", "alpaca", "sharegpt", "prompt_completion", "text"]


class ModelConfig(BaseModel):
    name: str = Field(
        description="Hugging Face model id or local path, e.g. Qwen/Qwen2.5-0.5B-Instruct"
    )
    load_in_4bit: bool = Field(
        False, description="QLoRA: load base model in 4-bit NF4 (needs bitsandbytes + CUDA)"
    )
    load_in_8bit: bool = Field(
        False, description="Load base model in 8-bit (needs bitsandbytes + CUDA)"
    )
    dtype: Literal["auto", "bfloat16", "float16", "float32"] = Field(
        "auto",
        description="Compute dtype. 'auto' picks bf16 on supported GPUs, fp16 on older GPUs, fp32 on CPU.",
    )
    attn_implementation: str | None = Field(
        None, description="e.g. 'flash_attention_2' or 'sdpa'. None lets transformers choose."
    )
    trust_remote_code: bool = False
    chat_template: str | None = Field(
        None, description="Override the tokenizer chat template (Jinja string). Rarely needed."
    )


class DataConfig(BaseModel):
    path: str = Field(
        description="Local file (.jsonl/.json/.csv/.parquet), directory, or Hub dataset id"
    )
    format: DataFormat = Field(
        "auto", description="Input schema. 'auto' detects from the first row."
    )
    split: str = Field("train", description="Split to use when loading from the Hub")
    subset: str | None = Field(None, description="Hub dataset config/subset name")
    eval_path: str | None = Field(None, description="Optional separate eval file / dataset id")
    eval_fraction: float = Field(
        0.02,
        ge=0.0,
        lt=1.0,
        description="Hold out this fraction of train for eval if eval_path is not set",
    )
    max_samples: int | None = Field(
        None, description="Truncate the training set (useful for smoke tests)"
    )
    system_prompt: str | None = Field(
        None, description="Prepend a system message to every conversation that lacks one"
    )
    max_length: int = Field(
        2048, ge=64, description="Max tokens per example; longer examples are truncated"
    )
    shuffle_seed: int | None = Field(
        42, description="Seed used to shuffle before splitting. None disables shuffling."
    )
    # Column names, only needed when auto-detection can't find them.
    text_field: str = "text"
    prompt_field: str = "prompt"
    completion_field: str = "completion"


class LoraConfigModel(BaseModel):
    enabled: bool = Field(
        True, description="Set false for full fine-tuning (needs a lot more VRAM)"
    )
    r: int = Field(16, ge=1, description="LoRA rank")
    alpha: int = Field(32, ge=1, description="LoRA alpha (scaling = alpha / r)")
    dropout: float = Field(0.05, ge=0.0, le=1.0)
    target_modules: str | list[str] = Field(
        "all-linear", description="'all-linear' or a list like [q_proj, k_proj, v_proj, o_proj]"
    )
    use_rslora: bool = False
    use_dora: bool = False
    modules_to_save: list[str] | None = Field(
        None,
        description="Extra modules to train fully, e.g. [embed_tokens, lm_head] when adding tokens",
    )


class TrainConfig(BaseModel):
    output_dir: str = "outputs/run"
    epochs: float = Field(1.0, gt=0)
    max_steps: int = Field(-1, description="Overrides epochs when > 0")
    batch_size: int = Field(2, ge=1, description="Per-device micro batch size")
    grad_accum: int = Field(
        8,
        ge=1,
        description="Gradient accumulation steps. Effective batch = batch_size * grad_accum * n_gpus",
    )
    lr: float = Field(2e-4, gt=0)
    scheduler: str = "cosine"
    warmup_ratio: float = Field(0.03, ge=0.0, le=1.0)
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optimizer: str = Field(
        "adamw_torch", description="e.g. adamw_torch, adamw_8bit, paged_adamw_8bit, adafactor"
    )
    gradient_checkpointing: bool = True
    packing: bool = Field(
        False,
        description="Pack multiple short examples into one sequence. Faster, but can hurt chat models.",
    )
    assistant_only_loss: bool = Field(
        False,
        description="Only compute loss on assistant turns. Needs a chat template with {% generation %} markers.",
    )
    logging_steps: int = 10
    eval_steps: int | None = Field(
        None, description="Evaluate every N steps. None = once per epoch."
    )
    save_steps: int | None = Field(
        None, description="Checkpoint every N steps. None = once per epoch."
    )
    save_total_limit: int = 2
    seed: int = 42
    report_to: list[str] = Field(
        default_factory=lambda: ["none"], description="e.g. [wandb], [tensorboard]"
    )
    run_name: str | None = None
    resume_from_checkpoint: str | bool = Field(
        False, description="Path to a checkpoint dir, or true to pick the latest"
    )
    dataloader_num_workers: int = 0
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Passed straight to SFTConfig (escape hatch)"
    )

    @field_validator("report_to", mode="before")
    @classmethod
    def _coerce_report_to(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        return v


class HubConfig(BaseModel):
    push: bool = False
    repo_id: str | None = Field(None, description="e.g. your-username/my-finetune")
    private: bool = True


class RunConfig(BaseModel):
    model: ModelConfig
    data: DataConfig
    lora: LoraConfigModel = Field(default_factory=LoraConfigModel)
    train: TrainConfig = Field(default_factory=TrainConfig)
    hub: HubConfig = Field(default_factory=HubConfig)

    # ----------------------------------------------------------------- I/O
    @classmethod
    def from_yaml(cls, path: str | Path, overrides: list[str] | None = None) -> RunConfig:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        if overrides:
            raw = apply_overrides(raw, overrides)
        return cls.model_validate(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], overrides: list[str] | None = None) -> RunConfig:
        if overrides:
            raw = apply_overrides(raw, overrides)
        return cls.model_validate(raw)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(self.model_dump(mode="json"), f, sort_keys=False)


# ---------------------------------------------------------------------------
# --set a.b.c=value overrides
# ---------------------------------------------------------------------------


def _parse_scalar(value: str) -> Any:
    """Parse a CLI string into the most specific YAML scalar (int/float/bool/null/list)."""
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def apply_overrides(raw: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Apply ``["train.lr=1e-4", "model.name=foo"]`` onto a nested dict (returns a copy)."""
    import copy

    out = copy.deepcopy(raw)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like section.key=value, got: {item!r}")
        key, _, value = item.partition("=")
        parts = key.strip().split(".")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {key!r}: {p!r} is not a mapping")
        node[parts[-1]] = _parse_scalar(value.strip())
    return out


EXAMPLE_CONFIG = """\
# tunekit run config. Every field is optional except model.name and data.path.
# Override anything from the CLI:  tunekit train config.yaml --set train.lr=1e-4

model:
  name: Qwen/Qwen2.5-0.5B-Instruct   # any causal LM on the Hub, or a local path
  load_in_4bit: false                 # true = QLoRA (needs bitsandbytes + NVIDIA GPU)

data:
  path: data/train.jsonl              # .jsonl/.json/.csv/.parquet, a directory, or a Hub dataset id
  format: auto                        # auto | messages | alpaca | sharegpt | prompt_completion | text
  eval_fraction: 0.02                 # hold-out for eval loss
  max_length: 2048                    # tokens per example
  # system_prompt: "You are a helpful assistant."

lora:
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: all-linear

train:
  output_dir: outputs/my-run
  epochs: 1
  batch_size: 2
  grad_accum: 8                       # effective batch = 16
  lr: 2.0e-4
  gradient_checkpointing: true
  logging_steps: 10
  report_to: [none]                   # or [wandb], [tensorboard]

hub:
  push: false
  # repo_id: your-username/my-finetune
"""
