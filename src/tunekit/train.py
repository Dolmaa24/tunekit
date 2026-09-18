"""The training loop: config -> data -> model -> TRL SFTTrainer -> saved adapter."""

from __future__ import annotations

import inspect
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

from . import __version__
from .config import RunConfig
from .data import NormalisedData, prepare
from .model import apply_lora, detect_hardware, load_model, load_tokenizer, trainable_summary

console = Console()


def build_sft_config(cfg: RunConfig, data: NormalisedData, bf16: bool, fp16: bool) -> Any:
    from trl import SFTConfig

    t = cfg.train
    params = inspect.signature(SFTConfig.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": t.output_dir,
        "num_train_epochs": t.epochs,
        "max_steps": t.max_steps,
        "per_device_train_batch_size": t.batch_size,
        "per_device_eval_batch_size": t.batch_size,
        "gradient_accumulation_steps": t.grad_accum,
        "learning_rate": t.lr,
        "lr_scheduler_type": t.scheduler,
        "warmup_ratio": t.warmup_ratio,
        "weight_decay": t.weight_decay,
        "max_grad_norm": t.max_grad_norm,
        "optim": t.optimizer,
        "gradient_checkpointing": t.gradient_checkpointing,
        "gradient_checkpointing_kwargs": {"use_reentrant": False}
        if t.gradient_checkpointing
        else None,
        "logging_steps": t.logging_steps,
        "save_total_limit": t.save_total_limit,
        "seed": t.seed,
        "report_to": t.report_to,
        "run_name": t.run_name or Path(t.output_dir).name,
        "bf16": bf16,
        "fp16": fp16,
        "dataloader_num_workers": t.dataloader_num_workers,
        "packing": t.packing,
        "save_strategy": "steps" if t.save_steps else "epoch",
        "logging_first_step": True,
        "remove_unused_columns": True,
        "disable_tqdm": False,
    }
    if t.save_steps:
        kwargs["save_steps"] = t.save_steps
    if data.eval is not None:
        kwargs["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = (
            "steps" if t.eval_steps else "epoch"
        )
        if t.eval_steps:
            kwargs["eval_steps"] = t.eval_steps
    # TRL renamed max_seq_length -> max_length in 0.20.
    kwargs["max_length" if "max_length" in params else "max_seq_length"] = cfg.data.max_length
    if data.kind == "text":
        kwargs["dataset_text_field"] = "text"
    if data.kind == "messages" and t.assistant_only_loss and "assistant_only_loss" in params:
        kwargs["assistant_only_loss"] = True
    kwargs.update(t.extra)
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return SFTConfig(**kwargs)


def run(cfg: RunConfig, dry_run: bool = False) -> Path:
    """Execute a full training run. Returns the adapter/model output directory."""
    t0 = time.time()
    hw = detect_hardware()
    console.rule("[bold]tunekit train")
    console.print(f"[dim]tunekit {__version__}[/]  hardware: {hw.summary}")

    # -- data ----------------------------------------------------------------
    data = prepare(cfg.data)
    n_eval = len(data.eval) if data.eval is not None else 0
    console.print(
        f"data: {len(data.train):,} train / {n_eval:,} eval  "
        f"[dim](source format: {data.source_format} -> {data.kind}, dropped {data.dropped})[/]"
    )
    if data.dropped:
        for reason, n in sorted(data.drop_reasons.items(), key=lambda kv: -kv[1])[:3]:
            console.print(f"  [yellow]dropped {n}[/]: {reason}")

    # -- model ---------------------------------------------------------------
    tokenizer = load_tokenizer(cfg.model)
    if data.kind == "messages" and tokenizer.chat_template is None:
        raise RuntimeError(
            f"{cfg.model.name} has no chat template but the data is conversational. "
            "Pick an instruct/chat variant of the model or set model.chat_template."
        )
    model, dtype = load_model(cfg.model, hw)
    quantised = cfg.model.load_in_4bit or cfg.model.load_in_8bit
    model = apply_lora(model, cfg.lora, quantised, cfg.train.gradient_checkpointing)
    trainable, total = trainable_summary(model)
    console.print(
        f"model: {cfg.model.name}  dtype={str(dtype).replace('torch.', '')}"
        f"{' 4-bit' if cfg.model.load_in_4bit else ' 8-bit' if cfg.model.load_in_8bit else ''}  "
        f"trainable {trainable / 1e6:.1f}M / {total / 1e6:.0f}M ({100 * trainable / total:.2f}%)"
    )

    eff_batch = cfg.train.batch_size * cfg.train.grad_accum * max(1, hw.n_gpus)
    steps_per_epoch = max(1, len(data.train) // eff_batch)
    total_steps = (
        cfg.train.max_steps if cfg.train.max_steps > 0 else int(steps_per_epoch * cfg.train.epochs)
    )
    console.print(
        f"schedule: effective batch {eff_batch}, ~{steps_per_epoch} steps/epoch, {total_steps} total steps, lr {cfg.train.lr}"
    )

    bf16 = dtype.is_floating_point and str(dtype).endswith("bfloat16")
    fp16 = str(dtype).endswith("float16") and not bf16
    sft_config = build_sft_config(cfg, data, bf16=bf16, fp16=fp16)

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_yaml(out_dir / "tunekit.yaml")

    if dry_run:
        console.print(
            "[green]dry run OK[/] – config, data and model all load. Nothing was trained."
        )
        return out_dir

    # -- train ---------------------------------------------------------------
    from trl import SFTTrainer

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=data.train,
        eval_dataset=data.eval,
        processing_class=tokenizer,
    )
    resume = cfg.train.resume_from_checkpoint or None
    result = trainer.train(resume_from_checkpoint=resume)

    # -- save ----------------------------------------------------------------
    trainer.save_model(str(out_dir))  # adapter (or full model when lora.enabled=false)
    tokenizer.save_pretrained(str(out_dir))
    metrics = dict(result.metrics)
    if data.eval is not None:
        metrics.update(trainer.evaluate())
    meta = {
        "tunekit_version": __version__,
        "base_model": cfg.model.name,
        "lora": cfg.lora.enabled,
        "kind": data.kind,
        "train_examples": len(data.train),
        "eval_examples": n_eval,
        "hardware": asdict(hw),
        "metrics": metrics,
        "wall_time_s": round(time.time() - t0, 1),
    }
    (out_dir / "tunekit.json").write_text(json.dumps(meta, indent=2, default=str))
    _write_model_card(out_dir, cfg, meta)

    loss = metrics.get("train_loss")
    eval_loss = metrics.get("eval_loss")
    console.print(
        f"[green]done[/] in {meta['wall_time_s']}s  train_loss={loss:.4f}"
        + (f"  eval_loss={eval_loss:.4f}" if eval_loss is not None else "")
    )
    console.print(f"saved to [bold]{out_dir}[/]   try it:  tunekit chat {out_dir}")

    if cfg.hub.push:
        if not cfg.hub.repo_id:
            raise ValueError("hub.push is true but hub.repo_id is not set")
        console.print(f"pushing to hub: {cfg.hub.repo_id}")
        trainer.model.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)
        tokenizer.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)
    return out_dir


def _write_model_card(out_dir: Path, cfg: RunConfig, meta: dict[str, Any]) -> None:
    m = meta["metrics"]
    lines = [
        "---",
        f"base_model: {cfg.model.name}",
        "library_name: peft" if cfg.lora.enabled else "library_name: transformers",
        "tags: [tunekit, lora, fine-tuned]" if cfg.lora.enabled else "tags: [tunekit, fine-tuned]",
        "---",
        "",
        f"# Fine-tune of `{cfg.model.name}`",
        "",
        f"Trained with [tunekit](https://github.com/Dolmaa24/tunekit) {meta['tunekit_version']}.",
        "",
        "| | |",
        "|---|---|",
        f"| Base model | `{cfg.model.name}` |",
        f"| Method | {'LoRA r=' + str(cfg.lora.r) + ', alpha=' + str(cfg.lora.alpha) if cfg.lora.enabled else 'full fine-tune'}"
        + (" (QLoRA 4-bit)" if cfg.model.load_in_4bit else "")
        + " |",
        f"| Train examples | {meta['train_examples']:,} |",
        f"| Epochs | {cfg.train.epochs} |",
        f"| Learning rate | {cfg.train.lr} |",
        f"| Effective batch | {cfg.train.batch_size * cfg.train.grad_accum} |",
        f"| Train loss | {m.get('train_loss', float('nan')):.4f} |",
    ]
    if "eval_loss" in m:
        lines.append(f"| Eval loss | {m['eval_loss']:.4f} |")
    lines += [
        "",
        "## Usage",
        "",
        "```bash",
        f"tunekit chat {cfg.train.output_dir}",
        "```",
        "",
        "The full run config is in `tunekit.yaml`; reproduce with `tunekit train tunekit.yaml`.",
    ]
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
