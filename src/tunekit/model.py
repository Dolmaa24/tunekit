"""Model + tokenizer loading, quantisation and LoRA wiring."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .config import LoraConfigModel, ModelConfig


@dataclass
class Hardware:
    device: str  # cuda | mps | cpu
    name: str
    bf16: bool
    fp16: bool
    vram_gb: float | None
    n_gpus: int

    @property
    def summary(self) -> str:
        mem = f", {self.vram_gb:.1f} GB" if self.vram_gb else ""
        gpus = f" x{self.n_gpus}" if self.n_gpus > 1 else ""
        return f"{self.device}: {self.name}{gpus}{mem} (bf16={'yes' if self.bf16 else 'no'})"


def detect_hardware() -> Hardware:
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        bf16 = torch.cuda.is_bf16_supported()
        return Hardware(
            device="cuda",
            name=props.name,
            bf16=bf16,
            fp16=True,
            vram_gb=props.total_memory / 1e9,
            n_gpus=torch.cuda.device_count(),
        )
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # bf16 on MPS is supported by recent torch; fp16 training is unstable there.
        return Hardware(
            device="mps", name="Apple Silicon (MPS)", bf16=True, fp16=False, vram_gb=None, n_gpus=1
        )
    return Hardware(device="cpu", name="CPU", bf16=False, fp16=False, vram_gb=None, n_gpus=0)


def resolve_dtype(cfg: ModelConfig, hw: Hardware) -> torch.dtype:
    if cfg.dtype == "bfloat16":
        return torch.bfloat16
    if cfg.dtype == "float16":
        return torch.float16
    if cfg.dtype == "float32":
        return torch.float32
    if hw.bf16:
        return torch.bfloat16
    if hw.fp16:
        return torch.float16
    return torch.float32


def bitsandbytes_available() -> bool:
    return importlib.util.find_spec("bitsandbytes") is not None


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(cfg.name, trust_remote_code=cfg.trust_remote_code)
    if cfg.chat_template:
        tok.chat_template = cfg.chat_template
    if tok.pad_token is None:
        # Most decoder-only models ship without a pad token. Reusing EOS is the usual fix,
        # but it makes the model learn to never emit EOS if labels aren't masked. TRL masks
        # padding in labels, so this is safe here.
        tok.pad_token = tok.eos_token
    return tok


def _quant_config(cfg: ModelConfig, dtype: torch.dtype) -> Any | None:
    if not (cfg.load_in_4bit or cfg.load_in_8bit):
        return None
    if not bitsandbytes_available():
        raise RuntimeError(
            "load_in_4bit/load_in_8bit need bitsandbytes: pip install 'tunekit[quant]' "
            "(Linux + NVIDIA GPU only). On Mac/CPU set load_in_4bit: false."
        )
    from transformers import BitsAndBytesConfig

    if cfg.load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return BitsAndBytesConfig(load_in_8bit=True)


def load_model(cfg: ModelConfig, hw: Hardware | None = None) -> tuple[PreTrainedModel, torch.dtype]:
    hw = hw or detect_hardware()
    dtype = resolve_dtype(cfg, hw)
    quant = _quant_config(cfg, dtype)
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "trust_remote_code": cfg.trust_remote_code,
    }
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    if quant is not None:
        kwargs["quantization_config"] = quant
        kwargs["device_map"] = "auto"
    elif hw.device == "cuda" and hw.n_gpus == 1:
        kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
    model.config.use_cache = False  # incompatible with gradient checkpointing
    return model, dtype


def apply_lora(
    model: PreTrainedModel, cfg: LoraConfigModel, quantised: bool, gradient_checkpointing: bool
) -> PreTrainedModel:
    if not cfg.enabled:
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        return model
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if quantised:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=gradient_checkpointing
        )
    elif gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.r,
        lora_alpha=cfg.alpha,
        lora_dropout=cfg.dropout,
        target_modules=cfg.target_modules,
        use_rslora=cfg.use_rslora,
        use_dora=cfg.use_dora,
        modules_to_save=cfg.modules_to_save,
    )
    return get_peft_model(model, lora)


def trainable_summary(model: PreTrainedModel) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
