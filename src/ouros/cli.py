"""Command-line interface for Ouros."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ouros.orchestrator import create_milestone1_orchestrator


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Ouros CLI."""

    parser = argparse.ArgumentParser(description="Run the Ouros Milestone 1 walking skeleton.")
    parser.add_argument("problem", help="Research problem statement.")
    parser.add_argument(
        "--tag",
        dest="domain_tags",
        action="append",
        default=[],
        help="Optional domain tag. Can be supplied multiple times.",
    )
    parser.add_argument(
        "--system-config",
        type=Path,
        default=None,
        help="Path to config/system.yaml.",
    )
    parser.add_argument(
        "--models-config",
        type=Path,
        default=None,
        help="Path to config/models.yaml.",
    )
    args = parser.parse_args(argv)

    orchestrator = create_milestone1_orchestrator(
        system_config_path=args.system_config,
        models_config_path=args.models_config,
    )
    result = orchestrator.run(args.problem, args.domain_tags)
    print(result.report)
    return 0 if result.run.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
