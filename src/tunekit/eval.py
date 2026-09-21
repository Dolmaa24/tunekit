"""Evaluate a fine-tune: held-out loss / perplexity and sample generations, optionally against the base model.

Loss is measured where it matters for each data kind:

* ``messages``          – on the final assistant turn only (the prompt is everything before it)
* ``prompt_completion`` – on the completion only
* ``text``              – on the whole sequence
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from rich.console import Console
from rich.table import Table

from .config import DataConfig
from .data import prepare
from .inference import base_model_of, is_adapter_dir, load_for_inference, stream_reply

console = Console()


def _split_prompt(tok: Any, row: dict[str, Any], kind: str) -> tuple[str, str]:
    """Return (prompt_text, full_text) so that loss is taken on full_text[len(prompt_text):]."""
    if kind == "messages":
        msgs = row["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        if not full.startswith(prompt):  # unusual template; fall back to whole-sequence loss
            prompt = ""
        return prompt, full
    if kind == "prompt_completion":
        return row["prompt"], row["prompt"] + row["completion"]
    return "", row["text"]


@torch.inference_mode()
def nll_over_rows(
    model: Any, tok: Any, rows: list[dict[str, Any]], kind: str, max_length: int
) -> tuple[float, int]:
    """Mean per-token negative log-likelihood over the scored region, and the token count."""
    total_nll, total_tokens = 0.0, 0
    for row in rows:
        prompt, full = _split_prompt(tok, row, kind)
        ids = tok(full, truncation=True, max_length=max_length, return_tensors="pt")["input_ids"]
        n_prompt = len(tok(prompt)["input_ids"]) if prompt else 0
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        if (labels != -100).sum() < 2:  # prompt filled the whole window; nothing to score
            continue
        ids = ids.to(model.device)
        logits = model(input_ids=ids).logits[:, :-1].float()
        targets = labels[:, 1:].to(model.device)
        total_nll += F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ).item()
        total_tokens += int((targets != -100).sum())
    if total_tokens == 0:
        raise RuntimeError("No tokens to score; every example was prompt-only after truncation")
    return total_nll / total_tokens, total_tokens


def _prompt_messages(row: dict[str, Any], kind: str) -> list[dict[str, str]] | None:
    if kind == "messages":
        return row["messages"][:-1]
    if kind == "prompt_completion":
        return [{"role": "user", "content": row["prompt"]}]
    return None


def _reference(row: dict[str, Any], kind: str) -> str:
    if kind == "messages":
        return row["messages"][-1]["content"]
    if kind == "prompt_completion":
        return row["completion"]
    return ""


@dataclass
class EvalResult:
    path: str
    base_model: str
    kind: str
    examples: int
    scored_tokens: int
    tuned_loss: float
    tuned_ppl: float
    base_loss: float | None = None
    base_ppl: float | None = None
    samples: list[dict[str, str]] = field(default_factory=list)
    wall_time_s: float = 0.0


def run_eval(
    path: str,
    data_cfg: DataConfig,
    compare_base: bool = True,
    n_samples: int = 3,
    max_examples: int = 100,
    max_new_tokens: int = 128,
) -> EvalResult:
    t0 = time.time()
    nd = prepare(data_cfg)
    rows_ds = nd.eval if nd.eval is not None and len(nd.eval) > 0 else nd.train
    rows = [rows_ds[i] for i in range(min(max_examples, len(rows_ds)))]
    which = "eval split" if rows_ds is nd.eval else "train split (no eval split available)"
    console.print(f"scoring {len(rows)} examples from the {which}  [dim]({nd.kind})[/]")

    model, tok = load_for_inference(path)
    tuned_loss, n_tok = nll_over_rows(model, tok, rows, nd.kind, data_cfg.max_length)
    samples: list[dict[str, str]] = []
    for row in rows[:n_samples]:
        msgs = _prompt_messages(row, nd.kind)
        if msgs is None:
            break
        reply = "".join(
            stream_reply(model, tok, msgs, max_new_tokens=max_new_tokens, temperature=0)
        )
        samples.append(
            {"prompt": msgs[-1]["content"], "reference": _reference(row, nd.kind), "tuned": reply}
        )

    base_name = base_model_of(path) if is_adapter_dir(path) else path
    result = EvalResult(
        path=path,
        base_model=base_name,
        kind=nd.kind,
        examples=len(rows),
        scored_tokens=n_tok,
        tuned_loss=round(tuned_loss, 4),
        tuned_ppl=round(math.exp(tuned_loss), 3),
        samples=samples,
    )

    if compare_base and is_adapter_dir(path):
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        base, btok = load_for_inference(base_name)
        base_loss, _ = nll_over_rows(base, btok, rows, nd.kind, data_cfg.max_length)
        result.base_loss = round(base_loss, 4)
        result.base_ppl = round(math.exp(base_loss), 3)
        for s, row in zip(samples, rows[: len(samples)], strict=True):
            msgs = _prompt_messages(row, nd.kind)
            s["base"] = "".join(
                stream_reply(base, btok, msgs, max_new_tokens=max_new_tokens, temperature=0)
            )

    result.wall_time_s = round(time.time() - t0, 1)
    return result


def print_result(r: EvalResult) -> None:
    table = Table(
        title=f"held-out loss on {r.examples} examples ({r.scored_tokens:,} scored tokens)"
    )
    table.add_column("model")
    table.add_column("loss", justify="right")
    table.add_column("perplexity", justify="right")
    if r.base_loss is not None:
        table.add_row(f"base: {r.base_model}", f"{r.base_loss:.4f}", f"{r.base_ppl:.3f}")
    table.add_row(f"tuned: {r.path}", f"{r.tuned_loss:.4f}", f"{r.tuned_ppl:.3f}")
    console.print(table)
    if r.base_loss is not None:
        delta = r.base_loss - r.tuned_loss
        verdict = "[green]lower[/]" if delta > 0 else "[red]higher[/]"
        console.print(f"tuned loss is {verdict} than base by {abs(delta):.4f} nats/token")
    for i, s in enumerate(r.samples):
        console.rule(f"[dim]sample {i}")
        _labelled("bold cyan", "prompt", s["prompt"])
        _labelled("bold", "reference", s["reference"])
        if "base" in s:
            _labelled("bold yellow", "base", s["base"])
        _labelled("bold green", "tuned", s["tuned"])


def _labelled(style: str, label: str, text: str) -> None:
    """Print a coloured label followed by user/model text with markup disabled (it may contain '[')."""
    console.print(f"[{style}]{label}:[/] ", end="")
    console.print(text.strip(), markup=False, highlight=False)


def save_result(r: EvalResult, out: Path) -> Path:
    out.write_text(json.dumps(asdict(r), indent=2))
    return out
