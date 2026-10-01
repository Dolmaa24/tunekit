from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from tunekit.config import EXAMPLE_CONFIG, RunConfig, apply_overrides


def test_example_config_is_valid():
    cfg = RunConfig.model_validate(yaml.safe_load(EXAMPLE_CONFIG))
    assert cfg.model.name.startswith("Qwen")
    assert cfg.train.grad_accum == 8


def test_minimal_config():
    cfg = RunConfig.from_dict({"model": {"name": "m"}, "data": {"path": "d"}})
    assert cfg.lora.r == 16 and cfg.train.epochs == 1.0 and cfg.hub.push is False


def test_overrides_parse_scalars():
    raw = {"model": {"name": "m"}, "data": {"path": "d"}}
    out = apply_overrides(
        raw, ["train.lr=1e-4", "model.load_in_4bit=true", "lora.r=8", "train.report_to=[wandb]"]
    )
    cfg = RunConfig.model_validate(out)
    assert cfg.train.lr == 1e-4
    assert cfg.model.load_in_4bit is True
    assert cfg.lora.r == 8
    assert cfg.train.report_to == ["wandb"]
    assert raw["model"] == {"name": "m"}  # original untouched


def test_override_bad_shape():
    with pytest.raises(ValueError, match="section.key=value"):
        apply_overrides({}, ["train.lr"])


def test_report_to_string_coerced():
    cfg = RunConfig.from_dict(
        {"model": {"name": "m"}, "data": {"path": "d"}, "train": {"report_to": "wandb"}}
    )
    assert cfg.train.report_to == ["wandb"]


def test_validation_errors_are_loud():
    with pytest.raises(ValidationError):
        RunConfig.from_dict({"model": {"name": "m"}, "data": {"path": "d", "eval_fraction": 1.5}})


def test_yaml_roundtrip(tmp_path):
    cfg = RunConfig.from_dict({"model": {"name": "m"}, "data": {"path": "d"}})
    cfg.to_yaml(tmp_path / "c.yaml")
    assert RunConfig.from_yaml(tmp_path / "c.yaml") == cfg


def test_version_is_single_sourced():
    """pyproject must not carry its own version: hatch reads it from tunekit.__version__.

    Two literals drift — a release once shipped metadata saying 0.1.1 while the CLI said 0.1.0.
    """
    import tomllib

    raw = tomllib.loads((Path(__file__).resolve().parent.parent / "pyproject.toml").read_text())
    assert "version" in raw["project"].get("dynamic", []), (
        "pyproject should declare a dynamic version"
    )
    assert "version" not in raw["project"], (
        "pyproject has a static version; it would drift from __version__"
    )
    assert raw["tool"]["hatch"]["version"]["path"] == "src/tunekit/__init__.py"
