"""Merge adapters into the base model and export to GGUF / Ollama."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

from .inference import base_model_of, is_adapter_dir

console = Console()


def merge(
    adapter_dir: str, output_dir: str, base: str | None = None, dtype: str = "bfloat16"
) -> Path:
    """Merge a LoRA adapter into its base model and save a standalone HF model."""
    if not is_adapter_dir(adapter_dir):
        raise RuntimeError(f"{adapter_dir} is not an adapter directory (no adapter_config.json)")
    from peft import PeftModel

    base = base or base_model_of(adapter_dir)
    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
        dtype
    ]
    console.print(f"merging [bold]{adapter_dir}[/] into [bold]{base}[/] ({dtype})")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch_dtype, device_map="cpu")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(adapter_dir).save_pretrained(out)
    console.print(f"[green]merged model saved to {out}[/]")
    return out


def find_llama_cpp_converter(explicit: str | None = None) -> tuple[Path, str]:
    """Locate llama.cpp's convert_hf_to_gguf.py and the interpreter to run it with.

    Prefers a ``.venv`` inside the llama.cpp checkout if one exists (for people who installed
    the converter's pinned requirements there); otherwise the current interpreter, which works
    with tunekit's own transformers/torch as long as ``sentencepiece`` and ``protobuf<5`` are
    installed (``pip install 'tunekit[gguf]'``).
    """
    candidates: list[Path] = []
    if explicit:
        p = Path(explicit).expanduser()
        candidates += [p, p / "convert_hf_to_gguf.py"]
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(Path(os.environ["LLAMA_CPP_DIR"]) / "convert_hf_to_gguf.py")
    candidates += [
        Path.home() / "llama.cpp" / "convert_hf_to_gguf.py",
        Path.cwd() / "llama.cpp" / "convert_hf_to_gguf.py",
    ]
    for c in candidates:
        if c.is_file():
            venv_py = c.parent / ".venv" / "bin" / "python"
            return c, (str(venv_py) if venv_py.exists() else sys.executable)
    raise FileNotFoundError(
        "Could not find llama.cpp's convert_hf_to_gguf.py. Either:\n"
        "  git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp\n"
        "  pip install 'tunekit[gguf]'\n"
        "or pass --llama-cpp /path/to/llama.cpp, or set LLAMA_CPP_DIR."
    )


def to_gguf(
    model_dir: str, output: str | None = None, quant: str = "q8_0", llama_cpp: str | None = None
) -> Path:
    """Convert a merged HF model directory to GGUF using llama.cpp's converter.

    ``quant`` is one of the converter's --outtype values: f32, f16, bf16, q8_0, tq1_0, tq2_0, auto.
    For smaller quants (q4_k_m etc.) run llama.cpp's ``llama-quantize`` on the q8_0/f16 output.
    """
    converter, python = find_llama_cpp_converter(llama_cpp)
    model_dir_p = Path(model_dir)
    if is_adapter_dir(model_dir_p):
        raise RuntimeError("Merge the adapter first: tunekit merge <adapter> <merged-dir>")
    out = Path(output) if output else model_dir_p / f"{model_dir_p.name}-{quant}.gguf"
    cmd = [
        sys.executable,
        str(converter),
        str(model_dir_p),
        "--outfile",
        str(out),
        "--outtype",
        quant,
    ]
    console.print("[dim]$ " + " ".join(cmd) + "[/]")
    subprocess.run(cmd, check=True)
    console.print(f"[green]GGUF written to {out}[/]")
    return out


def write_ollama_modelfile(
    gguf_path: str, model_dir: str, output: str | None = None, system: str | None = None
) -> Path:
    """Write an Ollama Modelfile next to the GGUF, using the tokenizer's chat template family if known."""
    gguf = Path(gguf_path)
    out = Path(output) if output else gguf.parent / "Modelfile"
    lines = [f"FROM {gguf.name}", ""]
    if system:
        lines += [f'SYSTEM """{system}"""', ""]
    lines += [
        "# Ollama reads the chat template embedded in the GGUF for most model families.",
        "# If replies look wrong, add a TEMPLATE block here matching your base model.",
        "PARAMETER temperature 0.7",
        "PARAMETER stop <|im_end|>",
        "PARAMETER stop <|eot_id|>",
        "PARAMETER stop <|end_of_text|>",
    ]
    out.write_text("\n".join(lines) + "\n")
    console.print(f"[green]Modelfile written to {out}[/]")
    name = Path(model_dir).name.lower().replace("_", "-")
    if shutil.which("ollama"):
        console.print(
            f"create it with:  [bold]ollama create {name} -f {out}[/]  then  ollama run {name}"
        )
    else:
        console.print(f"install Ollama (https://ollama.com) then:  ollama create {name} -f {out}")
    return out
