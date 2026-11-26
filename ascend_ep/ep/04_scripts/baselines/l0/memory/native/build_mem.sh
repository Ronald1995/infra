#!/usr/bin/env bash
# Build the local memory baselines (MEM-002 MTE copy, MEM-003 local UB copy).
#
#   bash build_mem.sh [BUILD_DIR] [SOC_TYPE]
#
# Requires ASCEND_HOME_PATH with the Ascend C kernel toolchain (bisheng).
# The library is optional: without it MEM-002/MEM-003 report dependency_missing.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${1:-${script_dir}/build}"
soc_type="${2:-Ascend950}"

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  printf 'ASCEND_HOME_PATH is not set; source the CANN set_env.sh first\n' >&2
  exit 2
fi

mkdir -p "${build_dir}"
"${ASCEND_HOME_PATH}/compiler/bisheng/bin/clang++" \
  --cce-aicore-arch=dav-c220 \
  -O2 -shared -fPIC -std=c++17 \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${ASCEND_HOME_PATH}/include/ascendc" \
  "${script_dir}/mem_kernel.cpp" \
  -o "${build_dir}/libl0_mem.so" \
  || {
    printf '\nThe exact kernel build flags depend on the CANN release. Use the\n' >&2
    printf 'Ascend C sample build system as a reference:\n' >&2
    printf '  ${ASCEND_HOME_PATH}/tools/tikcpp/tikcfw ... or aclnn custom-op sample\n' >&2
    printf 'Then set benchmark.memory_library=%s\n' "${build_dir}/libl0_mem.so" >&2
    exit 2
  }
printf 'built: %s\n' "${build_dir}/libl0_mem.so"
printf 'Set benchmark.memory_library to that path (or EP_L0_MEMORY_LIBRARY).\n'
