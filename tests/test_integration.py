"""End-to-end smoke test. Downloads SmolLM2-135M (~270 MB) and trains for a few steps on CPU/GPU.

Run with:  pytest -m integration
"""

from pathlib import Path

import pytest

from tunekit.config import RunConfig
from tunekit.train import run

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.integration
def test_train_chat_merge(tmp_path):
    cfg = RunConfig.from_dict(
        {
            "model": {"name": "HuggingFaceTB/SmolLM2-135M-Instruct", "dtype": "float32"},
            "data": {"path": str(EXAMPLES / "pirate.jsonl"), "max_length": 128, "max_samples": 16},
            "lora": {"r": 4, "alpha": 8},
            "train": {
                "output_dir": str(tmp_path / "run"),
                "max_steps": 3,
                "batch_size": 2,
                "grad_accum": 1,
                "gradient_checkpointing": False,
                "logging_steps": 1,
                "save_total_limit": 1,
            },
        }
    )
    out = run(cfg)
    assert (out / "adapter_model.safetensors").exists()
    assert (out / "tunekit.json").exists()
    assert (out / "README.md").exists()

    from tunekit.inference import load_for_inference, stream_reply

    model, tok = load_for_inference(str(out))
    reply = "".join(
        stream_reply(
            model, tok, [{"role": "user", "content": "hi"}], max_new_tokens=5, temperature=0
        )
    )
    assert isinstance(reply, str)

    from tunekit.export import merge

    merged = merge(str(out), str(tmp_path / "merged"), dtype="float32")
    assert (merged / "model.safetensors").exists()
    assert not (merged / "adapter_config.json").exists()
