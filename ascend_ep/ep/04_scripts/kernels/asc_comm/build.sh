#!/usr/bin/env bash
# Build the ASC-COMM communication kernels.
#
# Exits non-zero with the exact blocker when the CANN toolchain is absent.  It never
# pretends to have produced a kernel: the benchmark's capability probe reads
# build/manifest.json, and no manifest is written unless the compiler step succeeded.
#
# Usage:
#   ./build.sh [--soc Ascend950] [--aiv-only|--ccu-only] [--strict] [--clean]
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${here}/build"
soc="unknown"
engines_aiv="ON"
engines_ccu="ON"
strict="OFF"

while (($#)); do
  case "$1" in
    --soc) shift; soc="${1:?--soc needs a value}" ;;
    --aiv-only) engines_ccu="OFF" ;;
    --ccu-only) engines_aiv="OFF" ;;
    --strict) strict="ON" ;;
    --clean) rm -rf "${build_dir}" ;;
    -h|--help) sed -n '2,12p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

fail() {
  local status="$1"; shift
  echo "asc_comm build: BLOCKED" >&2
  echo "  reason: $*" >&2
  echo "  status: ${status}" >&2
  echo "  The AIV/CCU data planes stay unavailable; the benchmark will report" >&2
  echo "  dependency_missing or not_implemented for those cases. It will NOT fall" >&2
  echo "  back to the host NBI path." >&2
  exit 1
}

# ---------------------------------------------------------------- toolchain --
[[ -n "${ASCEND_HOME_PATH:-}" ]] || fail dependency_missing \
  "ASCEND_HOME_PATH is not set; source the CANN set_env.sh of the installed toolkit"

[[ -d "${ASCEND_HOME_PATH}" ]] || fail dependency_missing \
  "ASCEND_HOME_PATH points at a non-existent directory: ${ASCEND_HOME_PATH}"

command -v cmake >/dev/null 2>&1 || fail dependency_missing "cmake is not installed"
cxx="$(command -v g++ || command -v c++ || true)"
[[ -n "${cxx}" ]] || fail dependency_missing "no C++ compiler (g++/c++) is installed"

# The device sources are the part that needs the vendor headers.  Their absence is a
# real blocker, not a warning: without them the library would export only stubs that
# return UNSUPPORTED.
missing=()
[[ "${engines_aiv}" == "ON" && ! -f "${here}/aiv/write_nbi/kernel.cpp" ]] && \
  missing+=("aiv/write_nbi/kernel.cpp")
[[ "${engines_ccu}" == "ON" && ! -f "${here}/ccu/dsl/kernel.cpp" ]] && \
  missing+=("ccu/dsl/kernel.cpp")
if ((${#missing[@]})); then
  if [[ "${strict}" == "ON" ]]; then
    fail not_implemented "device sources are missing: ${missing[*]}"
  fi
  echo "asc_comm build: WARNING device sources missing: ${missing[*]}" >&2
  echo "  -> those engines will report ASC_COMM_ERR_UNSUPPORTED at run time" >&2
fi

cmake --version >/dev/null 2>&1 || fail dependency_missing "cmake is not runnable"

echo "asc_comm build: soc=${soc} aiv=${engines_aiv} ccu=${engines_ccu} strict=${strict}"
cmake -S "${here}" -B "${build_dir}" \
      -DASC_COMM_SOC_NAME="${soc}" \
      -DASC_COMM_BUILD_AIV="${engines_aiv}" \
      -DASC_COMM_BUILD_CCU="${engines_ccu}" \
      -DASC_COMM_STRICT="${strict}" \
      -DCMAKE_BUILD_TYPE=Release

cmake --build "${build_dir}" --parallel

[[ -f "${build_dir}/manifest.json" ]] || fail not_implemented \
  "the build finished but produced no manifest; the probe would be unable to verify it"

echo "asc_comm build: OK ${build_dir}/manifest.json"
