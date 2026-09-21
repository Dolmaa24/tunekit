import torch

from tunekit.config import ModelConfig
from tunekit.model import Hardware, hub_weight_bytes, resolve_dtype, weight_fit_report


def _cuda(vram_gb: float) -> Hardware:
    return Hardware(device="cuda", name="fake", bf16=True, fp16=True, vram_gb=vram_gb, n_gpus=1)


def test_hub_weight_bytes_local_dir(tmp_path):
    (tmp_path / "model-00001.safetensors").write_bytes(b"x" * 1000)
    (tmp_path / "model-00002.safetensors").write_bytes(b"x" * 500)
    (tmp_path / "config.json").write_text("{}")
    assert hub_weight_bytes(str(tmp_path)) == 1500


def test_hub_weight_bytes_local_dir_without_weights(tmp_path):
    assert hub_weight_bytes(str(tmp_path)) is None


def test_weight_fit_report_only_on_cuda(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"x" * 10)
    cpu = Hardware(device="cpu", name="CPU", bf16=False, fp16=False, vram_gb=None, n_gpus=0)
    assert weight_fit_report(str(tmp_path), cpu, False, False, torch.float32) is None


def test_weight_fit_report_verdicts(tmp_path):
    with open(tmp_path / "model.safetensors", "wb") as f:
        f.truncate(10_000_000_000)  # sparse 10 GB file: st_size is what matters, no bytes written
    # bf16 on a 16 GB card: 10 GB resident < 12.8 -> fits
    assert "fits" in weight_fit_report(str(tmp_path), _cuda(16), False, False, torch.bfloat16)
    # bf16 on an 8 GB card: 10 GB > 8 -> does not fit, and suggests 4-bit
    r = weight_fit_report(str(tmp_path), _cuda(8), False, False, torch.bfloat16)
    assert "does NOT fit" in r and "load_in_4bit" in r
    # 4-bit on the same card: ~3 GB resident -> fits
    assert "fits" in weight_fit_report(str(tmp_path), _cuda(8), True, False, torch.bfloat16)


def test_resolve_dtype_auto():
    assert resolve_dtype(ModelConfig(name="m"), _cuda(8)) == torch.bfloat16
    no_bf16 = Hardware(device="cuda", name="T4", bf16=False, fp16=True, vram_gb=15, n_gpus=1)
    assert resolve_dtype(ModelConfig(name="m"), no_bf16) == torch.float16
    cpu = Hardware(device="cpu", name="CPU", bf16=False, fp16=False, vram_gb=None, n_gpus=0)
    assert resolve_dtype(ModelConfig(name="m"), cpu) == torch.float32
    assert resolve_dtype(ModelConfig(name="m", dtype="float16"), cpu) == torch.float16
