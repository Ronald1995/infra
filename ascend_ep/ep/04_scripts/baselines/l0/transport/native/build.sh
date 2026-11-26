#!/usr/bin/env bash
# Build the optional L0 native transport bridge.
#
#   bash build.sh [BUILD_DIR]
#
# Requires a CANN installation with the HCOMM headers:
#   export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
#
# The bridge is optional.  When it is absent every LINK-* case reports
# `dependency_missing` and the rest of L0 keeps working.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  printf 'ASCEND_HOME_PATH is not set; source the CANN set_env.sh first\n' >&2
  exit 2
fi
if [[ ! -f "${ASCEND_HOME_PATH}/include/hcomm/hcomm_res.h" ]]; then
  printf 'HCOMM headers not found under %s/include/hcomm\n' "${ASCEND_HOME_PATH}" >&2
  printf 'Install the HCOMM component (CANN >= 9.1) or skip the native transport cases.\n' >&2
  exit 2
fi

cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel
printf '\nBuilt: %s\n' "$(find "${build_dir}" -name 'libl0_transport_bridge.so' -print -quit)"
printf 'Export it for the benchmark:\n  export EP_L0_TRANSPORT_LIBRARY=%s\n' \
  "$(find "${build_dir}" -name 'libl0_transport_bridge.so' -print -quit)"
