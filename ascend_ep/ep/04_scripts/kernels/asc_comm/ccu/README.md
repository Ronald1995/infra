# CCU data plane: an independent transfer kernel

## What "CCU" has to mean here

`LINK-008` is the CCU URMA case. Before this project it bound the HCOMM channel to
the CCU engine and then submitted through the host NBI path — the `engine` argument
changed, the data path did not. Changing a parameter is not a data plane.

A CCU case is only a CCU measurement when all of the following are true:

1. a CCU kernel is launched through the CCU frontend;
2. the kernel is bound to the same communication resources the channel acquired;
3. the kernel itself performs the remote-memory access;
4. completion is reported by the CCU scheduler, not by a host fence;
5. the profiler shows a CCU kernel record.

If any of these is missing, `LINK-008` is blocked. It is never served by the host
plane.

## Required toolchain artifacts (check these first)

- [ ] the CCU DSL frontend that ships with the installed CANN toolkit (compiler
      driver + include directory), and its version;
- [ ] the CCU kernel launch API and the handle type it takes;
- [ ] the binding API between an acquired HCOMM channel and a CCU communication
      resource;
- [ ] the CCU completion/status query API and its timeout semantics;
- [ ] the SoC capability note: CCU is not available on every Ascend generation, and
      on 950 only certain protocols expose it. `LINK-008` must therefore also be
      gated by the runtime capability probe (`channel_acquired` + `kernel_available`),
      not only by a compiled channel.

As with AIV, **no signature is invented in this repository**. The list above is the
input to writing `ccu/dsl/kernel.cpp`.

## Why `ccu/dsl/kernel.cpp` is not committed yet

The CCU DSL is the least portable part of the stack: its entry point, its resource
binding and its completion model are exactly what has to be read from the installed
toolkit. A guessed kernel would be unbuildable, unverifiable, and would let a reader
believe `LINK-008` had a real CCU path. It is reported as blocked instead.

```
kernels/asc_comm/ccu/dsl/kernel.cpp   <- to be written on the target machine
```

`CMakeLists.txt` picks it up automatically (`ASC_COMM_BUILD_CCU`), and
`build.sh --strict` fails while it is missing.

## Completion semantics

`asc_comm_ccu_wait` must block until the scheduler reports the remote writes retired.
`baselines/l0/transport/completion.py::CCU_COMPLETION` declares `remote_visible`, and
`tests/test_completion.py` asserts that a `submitted`/`fenced` primitive can never be
declared `remote_visible`. If the real CCU API turns out to report only submission,
the contract must be downgraded — and then `LINK-008` may not be used for bandwidth.

## Evidence required before `LINK-008` counts

* `channel_acquired` for `CcuUrmaChannel` (runtime probe, resources released after);
* `kernel_available`: `build/manifest.json` has a CCU entry and the library exports
  `asc_comm_ccu_submit` / `asc_comm_ccu_wait`;
* `kernel_verified`: msprof shows a CCU kernel record for the case;
* `execution_verified`: the element-wise payload check passes on both peers;
* `physical_link_verified`: the observed link comes from runtime evidence (channel
  kind + rank graph + msprof), never from the configured protocol.

Only then may a `LINK-008` row be reported as verified. See
`docs/evidence_capability.md` for the ladder.
