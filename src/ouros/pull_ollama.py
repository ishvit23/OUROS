"""Pull Ollama models referenced in config/models.yaml for local runs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from ouros.config import load_models_config


def list_ollama_model_tags(models: dict[str, str]) -> list[str]:
    """Return sorted unique Ollama image tags (without the `ollama/` prefix)."""

    tags: set[str] = set()
    for value in models.values():
        if isinstance(value, str) and value.startswith("ollama/"):
            tag = value.removeprefix("ollama/").strip()
            if tag:
                tags.add(tag)
    return sorted(tags)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for pulling Ollama models."""

    parser = argparse.ArgumentParser(
        description="Pull every ollama/* model listed in config/models.yaml.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Path to config/models.yaml (default: project config/models.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ollama pull commands without running them.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    models = load_models_config(args.models_config) if args.models_config else load_models_config()
    tags = list_ollama_model_tags(models)
    if not tags:
        print("No ollama/* models found in models config.", file=sys.stderr)
        return 0

    if args.dry_run:
        for tag in tags:
            print(f"ollama pull {tag}")
        return 0

    ollama = shutil.which("ollama")
    if ollama is None:
        print(
            "ollama is not on PATH. Install Ollama from https://ollama.com and ensure "
            "`ollama` is available, then re-run this command.",
            file=sys.stderr,
        )
        return 1

    for tag in tags:
        print(f"Pulling {tag} ...", flush=True)
        result = subprocess.run([ollama, "pull", tag], check=False)
        if result.returncode != 0:
            print(f"ollama pull {tag} failed with exit code {result.returncode}.", file=sys.stderr)
            return result.returncode

    print("All configured Ollama models are pulled.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
