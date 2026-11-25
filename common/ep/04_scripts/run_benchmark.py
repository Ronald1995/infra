"""Command-line entry point for the EP benchmark scaffold."""

import argparse
from collections.abc import Sequence
from pathlib import Path

from backends.registry import BACKEND_MODULES


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--list-backends",
        action="store_true",
        help="list the planned backend adapters",
    )
    actions.add_argument(
        "--config",
        type=Path,
        help="select a YAML configuration for a benchmark run",
    )
    args = parser.parse_args(argv)

    if args.list_backends:
        for backend, module in sorted(BACKEND_MODULES.items()):
            print(f"{backend}\t{module}")
        return 0

    if args.config is None:
        parser.print_help()
        return 0

    if not args.config.is_file():
        parser.error(f"configuration file does not exist: {args.config}")

    parser.exit(2, "Benchmark execution is not implemented in this scaffold.\n")


if __name__ == "__main__":
    raise SystemExit(main())
