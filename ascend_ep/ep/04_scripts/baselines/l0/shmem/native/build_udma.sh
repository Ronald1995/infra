#!/usr/bin/env bash
# Build the L0 SHMEM device kernels.
#
#   bash build_udma.sh /path/to/cann-shmem-checkout /path/to/build [SOC_TYPE]
#
# The SHMEM checkout must be the revision pinned below; the upstream build
# system is reused instead of re-implementing its CMake glue.
#
# Outputs:
#   <build>/build/examples/l0_udma/libep_l0_udma.so      (multi-outstanding kernel)
#   <build>/build/examples/l0_udma/libep_udma_kernel.so (legacy single-request ABI)
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: bash %s SHMEM_CHECKOUT BUILD_DIRECTORY [SOC_TYPE]\n' "$0" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(realpath "$1")"
build_root="$(realpath -m "$2")"
soc_type="${3:-Ascend950}"
revision=f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5

if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Expected CANN SHMEM %s (the API used here is pinned to it)\n' "$revision" >&2
  git -C "$source_root" rev-parse HEAD >&2
  exit 2
fi
mkdir -p -- "$build_root/shmem"
git -C "$source_root" archive "$revision" | tar -xf - -C "$build_root/shmem"
examples="$build_root/shmem/examples"
mkdir -p -- "$examples/l0_udma"
cp -- "$script_dir/udma_kernel.cpp" "$examples/l0_udma/"
cat > "$examples/l0_udma/CMakeLists.txt" <<'EOF'
set(USE_EXAMPLES OFF)
aclshmem_add_collective_example(l0_udma)
target_link_libraries(l0_udma_kernel PRIVATE shmem)
add_custom_command(TARGET l0_udma_kernel POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:l0_udma_kernel>
            $<TARGET_FILE_DIR:l0_udma_kernel>/libep_udma_kernel.so)
EOF
printf '\nadd_subdirectory(l0_udma)\n' >> "$examples/CMakeLists.txt"
cmake -S "$build_root/shmem" -B "$build_root/build" \
  -DSOC_TYPE="$soc_type" -DUSE_EXAMPLES=ON -DACLSHMEM_RDMA_SUPPORT=ON
cmake --build "$build_root/build" --target l0_udma_kernel --parallel
find "$build_root/build" -name 'libep_l0_udma.so' -o -name 'libep_udma_kernel.so' | while read -r lib; do
  printf 'built: %s\n' "$lib"
done
