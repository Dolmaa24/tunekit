"""Dataset loading, format detection and normalisation.

tunekit accepts the common instruction-tuning schemas and normalises them into
one of three canonical kinds that TRL's ``SFTTrainer`` understands natively:

* ``messages``          – ``{"messages": [{"role": ..., "content": ...}, ...]}``
* ``prompt_completion`` – ``{"prompt": str, "completion": str}`` (loss on completion only)
* ``text``              – ``{"text": str}`` (plain continued pre-training style)

Input schemas that are converted to ``messages``:

* alpaca    – ``instruction`` / ``input`` (optional) / ``output``
              (also accepts ``question``/``answer``, ``prompt``/``response``)
* sharegpt  – ``conversations: [{"from": "human"|"gpt"|"system", "value": ...}]``
* messages  – already canonical (OpenAI chat format); list-of-parts content is flattened
"""

from __future__ import annotations

import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset

from .config import DataConfig

SUPPORTED_EXTENSIONS = {".jsonl", ".json", ".csv", ".parquet", ".txt"}

_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
}
_VALID_ROLES = {"system", "user", "assistant", "tool"}

_ALPACA_INSTRUCTION_KEYS = ("instruction", "question", "prompt", "query")
_ALPACA_INPUT_KEYS = ("input", "context")
_ALPACA_OUTPUT_KEYS = ("output", "response", "answer", "completion")


class DataError(ValueError):
    """Raised for unusable datasets, with a message meant for end users."""


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _loader_for(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jsonl", ".json"}:
        return "json"
    if ext == ".csv":
        return "csv"
    if ext == ".parquet":
        return "parquet"
    if ext == ".txt":
        return "text"
    raise DataError(f"Unsupported file type {ext!r}. Use one of: {sorted(SUPPORTED_EXTENSIONS)}")


def load_raw(path: str, split: str = "train", subset: str | None = None) -> Dataset:
    """Load a local file, a directory of files, or a Hub dataset id into a ``Dataset``."""
    p = Path(os.path.expanduser(path))
    if p.is_file():
        ds = load_dataset(_loader_for(p), data_files=str(p), split="train")
    elif p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
        if not files:
            raise DataError(
                f"No data files found in {p}. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        kinds = {_loader_for(f) for f in files}
        if len(kinds) != 1:
            raise DataError(
                f"Directory {p} mixes file types {sorted(kinds)}; keep one type per directory."
            )
        ds = load_dataset(kinds.pop(), data_files=[str(f) for f in files], split="train")
    else:
        try:
            ds = load_dataset(path, subset, split=split)
        except Exception as e:  # noqa: BLE001 - surface a friendly message
            raise DataError(
                f"Could not load {path!r} as a file, directory or Hub dataset id.\n  {type(e).__name__}: {e}"
            ) from e
    if isinstance(ds, DatasetDict):
        ds = ds[split]
    return ds


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


def detect_format(example: dict[str, Any], cfg: DataConfig | None = None) -> str:
    """Guess the schema of one row. Returns one of the ``DataFormat`` literals (never 'auto')."""
    keys = set(example)
    if "messages" in keys and isinstance(example["messages"], list):
        return "messages"
    if "conversations" in keys and isinstance(example["conversations"], list):
        return "sharegpt"
    if {"chosen", "rejected"} <= keys:
        raise DataError(
            "This looks like a preference (DPO/ORPO) dataset with chosen/rejected columns. "
            "tunekit currently does supervised fine-tuning only."
        )
    pf = cfg.prompt_field if cfg else "prompt"
    cf = cfg.completion_field if cfg else "completion"
    if pf in keys and cf in keys and isinstance(example[pf], str) and isinstance(example[cf], str):
        return "prompt_completion"
    if any(k in keys for k in _ALPACA_INSTRUCTION_KEYS) and any(
        k in keys for k in _ALPACA_OUTPUT_KEYS
    ):
        return "alpaca"
    tf = cfg.text_field if cfg else "text"
    if tf in keys and isinstance(example[tf], str):
        return "text"
    raise DataError(
        "Could not detect the dataset format from columns "
        f"{sorted(keys)}.\nExpected one of:\n"
        "  messages:          {'messages': [{'role','content'}, ...]}\n"
        "  sharegpt:          {'conversations': [{'from','value'}, ...]}\n"
        "  alpaca:            {'instruction', 'input'?, 'output'}\n"
        "  prompt_completion: {'prompt', 'completion'}\n"
        "  text:              {'text'}\n"
        "Set data.format explicitly, or data.text_field / prompt_field / completion_field."
    )


# ---------------------------------------------------------------------------
# Conversion to canonical messages
# ---------------------------------------------------------------------------


def _first_present(d: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _flatten_content(content: Any) -> str:
    """OpenAI-style content can be a list of parts; keep the text ones."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type", "text") == "text":
                parts.append(str(part.get("text", "")))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def alpaca_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    instruction = _first_present(ex, _ALPACA_INSTRUCTION_KEYS)
    output = _first_present(ex, _ALPACA_OUTPUT_KEYS)
    if instruction is None or output is None:
        raise DataError(f"alpaca row is missing instruction/output: {list(ex)}")
    extra_input = _first_present(ex, _ALPACA_INPUT_KEYS)
    user = f"{instruction}\n\n{extra_input}" if extra_input else instruction
    msgs = []
    system = _first_present(ex, ("system", "system_prompt"))
    if system:
        msgs.append({"role": "system", "content": system})
    msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": output}]
    return msgs


def sharegpt_to_messages(ex: dict[str, Any]) -> list[dict[str, str]]:
    msgs = []
    system = _first_present(ex, ("system", "system_prompt"))
    if system:
        msgs.append({"role": "system", "content": system})
    for turn in ex["conversations"]:
        raw_role = str(turn.get("from", turn.get("role", ""))).lower()
        role = _SHAREGPT_ROLES.get(raw_role)
        if role is None:
            raise DataError(f"Unknown sharegpt speaker {raw_role!r}; expected human/gpt/system")
        msgs.append(
            {"role": role, "content": _flatten_content(turn.get("value", turn.get("content")))}
        )
    return msgs


def normalise_messages(msgs: list[dict[str, Any]]) -> list[dict[str, str]]:
    out = []
    for m in msgs:
        role = str(m.get("role", "")).lower()
        role = _SHAREGPT_ROLES.get(role, role)
        if role not in _VALID_ROLES:
            raise DataError(f"Invalid role {role!r}; expected one of {sorted(_VALID_ROLES)}")
        out.append({"role": role, "content": _flatten_content(m.get("content"))})
    return out


def validate_messages(msgs: list[dict[str, str]]) -> None:
    if not msgs:
        raise DataError("Empty conversation")
    if not any(m["role"] == "assistant" for m in msgs):
        raise DataError("Conversation has no assistant turn; nothing to learn from")
    if msgs[-1]["role"] != "assistant":
        raise DataError("Conversation should end with an assistant turn")
    for i, m in enumerate(msgs):
        if m["role"] == "system" and i != 0:
            raise DataError("System message must be the first turn")


@dataclass
class NormalisedData:
    kind: str  # messages | prompt_completion | text
    source_format: str
    train: Dataset
    eval: Dataset | None = None
    dropped: int = 0
    drop_reasons: dict[str, int] = field(default_factory=dict)


def _convert_row(ex: dict[str, Any], fmt: str, system_prompt: str | None) -> dict[str, Any]:
    if fmt == "messages":
        msgs = normalise_messages(ex["messages"])
    elif fmt == "alpaca":
        msgs = alpaca_to_messages(ex)
    elif fmt == "sharegpt":
        msgs = sharegpt_to_messages(ex)
    else:
        raise ValueError(fmt)
    if system_prompt and (not msgs or msgs[0]["role"] != "system"):
        msgs = [{"role": "system", "content": system_prompt}, *msgs]
    validate_messages(msgs)
    return {"messages": msgs}


def normalise(
    ds: Dataset, cfg: DataConfig, fmt: str | None = None
) -> tuple[Dataset, str, str, dict[str, int]]:
    """Convert a raw dataset to a canonical one. Returns (dataset, kind, source_format, drop_reasons)."""
    if len(ds) == 0:
        raise DataError("Dataset is empty")
    fmt = fmt or (detect_format(ds[0], cfg) if cfg.format == "auto" else cfg.format)

    if fmt == "text":
        col = cfg.text_field
        if col not in ds.column_names:
            raise DataError(f"text_field {col!r} not in columns {ds.column_names}")
        ds = ds.rename_column(col, "text") if col != "text" else ds
        ds = ds.select_columns(["text"]).filter(lambda r: bool(r["text"] and r["text"].strip()))
        return ds, "text", fmt, {}

    if fmt == "prompt_completion":
        pf, cf = cfg.prompt_field, cfg.completion_field
        for col in (pf, cf):
            if col not in ds.column_names:
                raise DataError(f"column {col!r} not in columns {ds.column_names}")
        ds = ds.select_columns([pf, cf])
        if pf != "prompt":
            ds = ds.rename_column(pf, "prompt")
        if cf != "completion":
            ds = ds.rename_column(cf, "completion")
        if cfg.system_prompt:
            sp = cfg.system_prompt
            ds = ds.map(lambda r: {"prompt": f"{sp}\n\n{r['prompt']}"})
        ds = ds.filter(lambda r: bool(r["completion"] and str(r["completion"]).strip()))
        return ds, "prompt_completion", fmt, {}

    # Conversational formats -> messages. Rows that fail validation are dropped, not fatal.
    reasons: dict[str, int] = {}
    converted: list[dict[str, Any]] = []
    for ex in ds:
        try:
            converted.append(_convert_row(ex, fmt, cfg.system_prompt))
        except DataError as e:
            reasons[str(e)] = reasons.get(str(e), 0) + 1
    if not converted:
        top = max(reasons, key=reasons.get) if reasons else "unknown"
        raise DataError(f"Every row was rejected. Most common reason: {top}")
    out = Dataset.from_list(converted)
    return out, "messages", fmt, reasons


def prepare(cfg: DataConfig) -> NormalisedData:
    """Load, normalise and split according to ``cfg``."""
    raw = load_raw(cfg.path, split=cfg.split, subset=cfg.subset)
    if cfg.shuffle_seed is not None:
        raw = raw.shuffle(seed=cfg.shuffle_seed)
    if cfg.max_samples:
        raw = raw.select(range(min(cfg.max_samples, len(raw))))
    train, kind, fmt, reasons = normalise(raw, cfg)
    dropped = sum(reasons.values())

    eval_ds: Dataset | None = None
    if cfg.eval_path:
        raw_eval = load_raw(cfg.eval_path, split=cfg.split, subset=cfg.subset)
        eval_ds, ekind, _, ereasons = normalise(raw_eval, cfg, fmt=fmt)
        if ekind != kind:
            raise DataError(f"eval set is {ekind} but train set is {kind}")
        dropped += sum(ereasons.values())
    elif cfg.eval_fraction > 0 and len(train) >= 20:
        n_eval = max(1, int(round(len(train) * cfg.eval_fraction)))
        split = train.train_test_split(test_size=n_eval, seed=cfg.shuffle_seed or 0, shuffle=False)
        train, eval_ds = split["train"], split["test"]

    return NormalisedData(
        kind=kind,
        source_format=fmt,
        train=train,
        eval=eval_ds,
        dropped=dropped,
        drop_reasons=reasons,
    )


# ---------------------------------------------------------------------------
# Stats (used by `tunekit validate`)
# ---------------------------------------------------------------------------


def render_example(tokenizer: Any, row: dict[str, Any], kind: str) -> str:
    """Return the exact string the model will be trained on for one row."""
    if kind == "messages":
        return tokenizer.apply_chat_template(row["messages"], tokenize=False)
    if kind == "prompt_completion":
        return row["prompt"] + row["completion"]
    return row["text"]


def token_stats(
    tokenizer: Any, ds: Dataset, kind: str, max_length: int, sample: int = 2000
) -> dict[str, Any]:
    n = min(len(ds), sample)
    rows = ds.select(range(n)) if n < len(ds) else ds
    lengths = [len(tokenizer(render_example(tokenizer, r, kind))["input_ids"]) for r in rows]
    lengths.sort()

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * len(lengths)))]

    return {
        "sampled": n,
        "total": len(ds),
        "min": lengths[0],
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p99": pct(0.99),
        "max": lengths[-1],
        "mean": round(statistics.fmean(lengths), 1),
        "over_max_length": sum(1 for x in lengths if x > max_length),
        "total_tokens_est": int(statistics.fmean(lengths) * len(ds)),
    }
