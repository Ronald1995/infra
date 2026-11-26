# Local memory baselines

| Case | What moves | Implementation |
| --- | --- | --- |
| MEM-001 | HBM device copy (contiguous) | `device.py`, torch `copy_` |
| MEM-002 | GM -> UB -> GM through MTE2/MTE3 | `local.py` + `native/mem_kernel.cpp` mode 0 |
| MEM-003 | UB -> UB inside one core | `local.py` + `native/mem_kernel.cpp` mode 1 |
| MEM-004 | HBM copy with stride 2 | `device.py`, same byte count as MEM-001 |

`MEM-002`/`MEM-003` need `libl0_mem.so` (`native/build_mem.sh`) and
`benchmark.memory_library`; without them they report `dependency_missing`.

These are local copies.  A local MTE or UB result is never reported as UB Memory
remote bandwidth.  Device copy reports logical read+write bytes; physical HBM
counter bytes stay empty when no counter is available.