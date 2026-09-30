"""Load a trained adapter (or merged model) and talk to it."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from threading import Thread
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .model import detect_hardware


def is_adapter_dir(path: str | Path) -> bool:
    return (Path(path) / "adapter_config.json").exists()


def base_model_of(adapter_dir: str | Path) -> str:
    cfg = json.loads((Path(adapter_dir) / "adapter_config.json").read_text())
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise RuntimeError(f"{adapter_dir}/adapter_config.json has no base_model_name_or_path")
    return base


def _quantization_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    from transformers import BitsAndBytesConfig

    return {
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        "device_map": "auto",
    }


def load_for_inference(
    path: str,
    base: str | None = None,
    dtype: torch.dtype | None = None,
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load a merged model dir / Hub id, or an adapter dir (base is read from adapter_config).

    ``load_in_4bit`` reloads the base model quantised, which is what you want for an adapter
    trained with QLoRA: the full-precision base of a 7B model needs ~15 GB, while the 4-bit
    base it was trained against needs ~5 GB and fits the same GPU.
    """
    hw = detect_hardware()
    dtype = dtype or (torch.bfloat16 if hw.bf16 else torch.float32)
    device = hw.device if hw.device != "cpu" else None
    quant = _quantization_kwargs(dtype) if load_in_4bit else {}
    if quant:
        device = None  # device_map="auto" already places the weights

    if is_adapter_dir(path):
        from peft import PeftModel

        base = base or base_model_of(path)
        tok = AutoTokenizer.from_pretrained(path)  # tunekit saves the tokenizer next to the adapter
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, **quant)
        try:
            model = PeftModel.from_pretrained(model, path)
        except ImportError as e:  # peft rejects an outdated optional integration
            if "torchao" in str(e):
                raise RuntimeError(
                    f"peft refused to load because of an incompatible torchao in this environment.\n"
                    f"  {e}\n"
                    "tunekit does not use torchao. Remove it (`pip uninstall -y torchao`) or upgrade it "
                    "(`pip install -U torchao`), then retry. Colab preinstalls an old torchao, which is "
                    "the usual cause."
                ) from e
            raise
    else:
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, **quant)
    if device:
        model.to(device)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


@torch.inference_mode()
def stream_reply(
    model: Any,
    tok: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Iterator[str]:
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs: dict[str, Any] = dict(
        **inputs, streamer=streamer, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id
    )
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
    else:
        gen_kwargs["do_sample"] = False
    thread = Thread(target=model.generate, kwargs=gen_kwargs)
    thread.start()
    yield from streamer
    thread.join()


def chat_loop(
    path: str,
    system: str | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    load_in_4bit: bool = False,
) -> None:
    from rich.console import Console

    console = Console()
    console.print(f"[dim]loading {path} ...[/]")
    model, tok = load_for_inference(path, load_in_4bit=load_in_4bit)
    console.print("[green]ready[/]. Type a message; /reset clears history, /quit exits.\n")
    history: list[dict[str, str]] = [{"role": "system", "content": system}] if system else []
    while True:
        try:
            user = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit", "/q"}:
            break
        if user == "/reset":
            history = history[:1] if system else []
            console.print("[dim]history cleared[/]")
            continue
        history.append({"role": "user", "content": user})
        console.print("[bold magenta]bot>[/] ", end="")
        reply = ""
        for piece in stream_reply(
            model, tok, history, max_new_tokens=max_new_tokens, temperature=temperature
        ):
            console.print(piece, end="", highlight=False, markup=False)
            reply += piece
        console.print()
        history.append({"role": "assistant", "content": reply})
