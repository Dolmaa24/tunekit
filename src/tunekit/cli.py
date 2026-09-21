"""tunekit command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .config import EXAMPLE_CONFIG, RunConfig

app = typer.Typer(
    name="tunekit",
    help="One-command LoRA/QLoRA fine-tuning for open LLMs.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_show_locals=False,
)
console = Console()

SetOpt = Annotated[
    list[str] | None,
    typer.Option(
        "--set", "-s", help="Override a config value, e.g. --set train.lr=1e-4", show_default=False
    ),
]


def _version(value: bool) -> None:
    if value:
        console.print(f"tunekit {__version__}")
        raise typer.Exit()


@app.callback()
def _main(
    version: Annotated[
        bool, typer.Option("--version", "-V", callback=_version, is_eager=True, help="Show version")
    ] = False,
) -> None:
    pass


def _load_config(
    config: Path | None,
    model: str | None,
    data: str | None,
    output: str | None,
    overrides: list[str] | None,
) -> RunConfig:
    overrides = list(overrides or [])
    if model:
        overrides.append(f"model.name={model}")
    if data:
        overrides.append(f"data.path={data}")
    if output:
        overrides.append(f"train.output_dir={output}")
    if config:
        return RunConfig.from_yaml(config, overrides)
    if not (model and data):
        raise typer.BadParameter("Pass a config file, or both --model and --data.")
    return RunConfig.from_dict({"model": {"name": model}, "data": {"path": data}}, overrides)


# ---------------------------------------------------------------------------


@app.command()
def init(
    path: Annotated[Path, typer.Argument(help="Where to write the starter config")] = Path(
        "config.yaml"
    ),
    force: Annotated[bool, typer.Option("--force", "-f", help="Overwrite if it exists")] = False,
) -> None:
    """Write a commented starter config you can edit."""
    if path.exists() and not force:
        console.print(f"[red]{path} exists[/] (use --force to overwrite)")
        raise typer.Exit(1)
    path.write_text(EXAMPLE_CONFIG)
    console.print(f"[green]wrote {path}[/]  next:  edit it, then  tunekit train {path}")


@app.command()
def validate(
    config: Annotated[
        Path | None, typer.Argument(help="Run config (optional if --data is given)")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model id (for tokenizer stats)")
    ] = None,
    data: Annotated[str | None, typer.Option("--data", "-d", help="Dataset path or Hub id")] = None,
    overrides: SetOpt = None,
    show: Annotated[int, typer.Option(help="Print this many rendered examples")] = 1,
) -> None:
    """Check a dataset: detect its format, convert it, and report token-length stats."""
    from .data import DataError, prepare, render_example, token_stats
    from .model import load_tokenizer

    model = model or "Qwen/Qwen2.5-0.5B-Instruct"
    cfg = _load_config(config, model, data, None, overrides)
    try:
        nd = prepare(cfg.data)
    except DataError as e:
        console.print(f"[red]data error:[/] {e}")
        raise typer.Exit(1) from None

    console.print(f"format: [bold]{nd.source_format}[/] -> {nd.kind}")
    n_eval = len(nd.eval) if nd.eval is not None else 0
    console.print(f"rows: {len(nd.train):,} train / {n_eval:,} eval / {nd.dropped} dropped")
    for reason, n in sorted(nd.drop_reasons.items(), key=lambda kv: -kv[1])[:5]:
        console.print(f"  [yellow]{n:>6}[/]  {reason}")

    tok = load_tokenizer(cfg.model)
    if nd.kind == "messages" and tok.chat_template is None:
        console.print(f"[red]{cfg.model.name} has no chat template; use an instruct/chat model[/]")
        raise typer.Exit(1)
    st = token_stats(tok, nd.train, nd.kind, cfg.data.max_length)
    table = Table(
        title=f"token lengths (tokenizer: {cfg.model.name}, sampled {st['sampled']:,}/{st['total']:,})"
    )
    for k in ("min", "p50", "p90", "p99", "max", "mean"):
        table.add_column(k, justify="right")
    table.add_row(*(str(st[k]) for k in ("min", "p50", "p90", "p99", "max", "mean")))
    console.print(table)
    console.print(f"~{st['total_tokens_est']:,} tokens per epoch")
    if st["over_max_length"]:
        pct = 100 * st["over_max_length"] / st["sampled"]
        console.print(
            f"[yellow]{st['over_max_length']} of {st['sampled']} sampled rows ({pct:.1f}%) exceed "
            f"max_length={cfg.data.max_length} and will be truncated.[/] Raise data.max_length or shorten them."
        )
    for i in range(min(show, len(nd.train))):
        console.rule(f"[dim]example {i}")
        console.print(render_example(tok, nd.train[i], nd.kind), markup=False, highlight=False)
    console.print("[green]dataset OK[/]")


@app.command()
def train(
    config: Annotated[
        Path | None, typer.Argument(help="YAML run config (see `tunekit init`)")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Base model id or path")
    ] = None,
    data: Annotated[str | None, typer.Option("--data", "-d", help="Dataset path or Hub id")] = None,
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output directory")] = None,
    overrides: SetOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Load everything, train nothing")
    ] = False,
) -> None:
    """Fine-tune a model. Use a config file, or --model + --data for defaults."""
    from .data import DataError
    from .train import run

    cfg = _load_config(config, model, data, output, overrides)
    try:
        run(cfg, dry_run=dry_run)
    except DataError as e:
        console.print(f"[red]data error:[/] {e}")
        raise typer.Exit(1) from None


@app.command()
def chat(
    path: Annotated[str, typer.Argument(help="Adapter dir, merged model dir, or Hub id")],
    system: Annotated[str | None, typer.Option("--system", help="System prompt")] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Max tokens per reply")] = 512,
    temperature: Annotated[float, typer.Option(help="0 = greedy")] = 0.7,
) -> None:
    """Chat interactively with a fine-tuned model."""
    from .inference import chat_loop

    chat_loop(path, system=system, max_new_tokens=max_new_tokens, temperature=temperature)


@app.command()
def merge(
    adapter: Annotated[str, typer.Argument(help="Adapter directory from `tunekit train`")],
    output: Annotated[str, typer.Argument(help="Where to write the merged model")],
    base: Annotated[
        str | None, typer.Option(help="Override base model (default: from adapter_config.json)")
    ] = None,
    dtype: Annotated[str, typer.Option(help="bfloat16 | float16 | float32")] = "bfloat16",
) -> None:
    """Merge a LoRA adapter into its base model -> standalone Hugging Face model."""
    from .export import merge as _merge

    _merge(adapter, output, base=base, dtype=dtype)


@app.command()
def export(
    model_dir: Annotated[
        str, typer.Argument(help="Merged model directory (run `tunekit merge` first)")
    ],
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output .gguf path")] = None,
    quant: Annotated[str, typer.Option(help="f16 | bf16 | q8_0 | f32")] = "q8_0",
    llama_cpp: Annotated[str | None, typer.Option(help="Path to a llama.cpp checkout")] = None,
    ollama: Annotated[
        bool, typer.Option("--ollama", help="Also write an Ollama Modelfile")
    ] = False,
    system: Annotated[
        str | None, typer.Option(help="System prompt to bake into the Modelfile")
    ] = None,
) -> None:
    """Export a merged model to GGUF (llama.cpp / Ollama / LM Studio)."""
    from .export import to_gguf, write_ollama_modelfile

    try:
        gguf = to_gguf(model_dir, output=output, quant=quant, llama_cpp=llama_cpp)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/]")
        raise typer.Exit(1) from None
    if ollama:
        write_ollama_modelfile(str(gguf), model_dir, system=system)


@app.command()
def eval(  # noqa: A001 - typer command name
    path: Annotated[
        str, typer.Argument(help="Adapter dir from `tunekit train`, merged dir, or Hub id")
    ],
    data: Annotated[
        str | None,
        typer.Option(
            "--data", "-d", help="Dataset path or Hub id (default: the run's tunekit.yaml data)"
        ),
    ] = None,
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Run config to take data settings from")
    ] = None,
    overrides: SetOpt = None,
    compare_base: Annotated[
        bool,
        typer.Option("--compare-base/--no-compare-base", help="Also score the untuned base model"),
    ] = True,
    samples: Annotated[int, typer.Option(help="Sample generations to show")] = 3,
    max_examples: Annotated[int, typer.Option(help="Cap on examples scored")] = 100,
    max_new_tokens: Annotated[int, typer.Option(help="Tokens per sample generation")] = 128,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Where to write eval.json (default: <path>/eval.json)"),
    ] = None,
) -> None:
    """Held-out loss / perplexity and sample generations, tuned vs base."""
    from .data import DataError
    from .eval import print_result, run_eval, save_result

    run_yaml = Path(path) / "tunekit.yaml"
    if config is None and data is None and run_yaml.exists():
        config = run_yaml
    if config is None and data is None:
        raise typer.BadParameter(
            "Pass --data, or --config, or a run directory containing tunekit.yaml."
        )
    cfg = _load_config(config, "unused" if config is None else None, data, None, overrides)
    try:
        result = run_eval(
            path,
            cfg.data,
            compare_base=compare_base,
            n_samples=samples,
            max_examples=max_examples,
            max_new_tokens=max_new_tokens,
        )
    except DataError as e:
        console.print(f"[red]data error:[/] {e}")
        raise typer.Exit(1) from None
    print_result(result)
    out = output or (Path(path) / "eval.json" if Path(path).is_dir() else Path("eval.json"))
    save_result(result, out)
    console.print(f"[dim]written to {out}[/]")


@app.command()
def push(
    path: Annotated[str, typer.Argument(help="Adapter or merged model directory")],
    repo_id: Annotated[str, typer.Argument(help="Hub repo, e.g. your-username/my-finetune")],
    private: Annotated[bool, typer.Option("--private/--public")] = True,
    message: Annotated[str, typer.Option("--message", "-m")] = "Upload from tunekit",
) -> None:
    """Upload a run directory to the Hugging Face Hub (needs `huggingface-cli login` or HF_TOKEN)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    src = Path(path)
    if not src.is_dir():
        console.print(f"[red]{src} is not a directory[/]")
        raise typer.Exit(1)
    api = HfApi()
    try:
        api.create_repo(repo_id, private=private, exist_ok=True)
        url = api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            commit_message=message,
            ignore_patterns=["checkpoint-*", "runs/*", "wandb/*"],
        )
    except HfHubHTTPError as e:
        console.print(f"[red]hub error:[/] {e}")
        console.print("Log in with:  huggingface-cli login   (or set HF_TOKEN)")
        raise typer.Exit(1) from None
    console.print(f"[green]uploaded[/] {src} -> https://huggingface.co/{repo_id}  [dim]{url}[/]")


@app.command()
def info(
    path: Annotated[str | None, typer.Argument(help="A run output directory to summarise")] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Check whether a model's weights fit this GPU"),
    ] = None,
) -> None:
    """Show detected hardware, whether a model fits it, or summarise a finished run."""
    from .model import bitsandbytes_available, detect_hardware, hub_weight_bytes

    hw = detect_hardware()
    console.print(f"tunekit {__version__}")
    console.print(f"hardware: {hw.summary}")
    console.print(
        f"bitsandbytes (QLoRA): {'available' if bitsandbytes_available() else 'not installed'}"
    )
    if model:
        size = hub_weight_bytes(model)
        if size is None:
            console.print(
                f"[yellow]could not read weight sizes for {model} (offline, gated, or not a model repo)[/]"
            )
        else:
            gb = size / 1e9
            console.print(f"{model}: {gb:.1f} GB of weights on disk")
            console.print(
                f"  resident base weights: bf16 ~{gb:.1f} GB | 8-bit ~{gb * 0.55:.1f} GB | 4-bit ~{gb * 0.30:.1f} GB  "
                "[dim](LoRA params, optimizer state and activations come on top)[/]"
            )
            if hw.vram_gb:
                for label, ratio in (("bf16", 1.0), ("8-bit", 0.55), ("4-bit", 0.30)):
                    need = gb * ratio
                    ok = (
                        "fits"
                        if need < hw.vram_gb * 0.8
                        else "tight"
                        if need < hw.vram_gb
                        else "no"
                    )
                    console.print(f"  {label:>5}: {ok}")
    if path:
        meta_path = Path(path) / "tunekit.json"
        if not meta_path.exists():
            console.print(f"[red]{meta_path} not found[/]")
            raise typer.Exit(1)
        meta = json.loads(meta_path.read_text())
        console.print_json(json.dumps(meta))


if __name__ == "__main__":
    app()
