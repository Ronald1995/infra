#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
  printf 'Usage: bash %s SHMEM_CHECKOUT NEW_BUILD_DIRECTORY\n' "$0" >&2
  exit 2
fi
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(realpath "$1")"
build_root="$(realpath -m "$2")"
revision=f0bcfc0fe51bce1d866ba6935d2adc6e5e5800e5
if [[ "$(git -C "$source_root" rev-parse HEAD)" != "$revision" ]]; then
  printf 'Expected CANN SHMEM %s\n' "$revision" >&2
  exit 2
fi
mkdir -- "$build_root"
mkdir -- "$build_root/shmem"
git -C "$source_root" archive "$revision" | tar -xf - -C "$build_root/shmem"
examples="$build_root/shmem/examples"
mkdir -- "$examples/ep_udma"
cp -- "$script_dir/ep_udma_kernel.cpp" "$examples/ep_udma/"
cat > "$examples/ep_udma/CMakeLists.txt" <<'EOF'
set(USE_EXAMPLES OFF)
aclshmem_add_collective_example(ep_udma)
target_link_libraries(ep_udma_kernel PRIVATE shmem)
EOF
printf '\nadd_subdirectory(ep_udma)\n' >> "$examples/CMakeLists.txt"
cmake -S "$build_root/shmem" -B "$build_root/build" \
  -DSOC_TYPE=Ascend950 -DUSE_EXAMPLES=ON -DACLSHMEM_RDMA_SUPPORT=ON
cmake --build "$build_root/build" --target ep_udma_kernel --parallel
find "$build_root/build" -name 'libep_udma_kernel.so' -print
