#!/usr/bin/env bash
set -euo pipefail

scripts_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec python3 "$scripts_dir/run_benchmark.py" --config "$scripts_dir/configs/ep128_bf16.yaml" "$@"
