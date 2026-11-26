# AIV data plane: `write_nbi` + drain

## What the kernel must do

For one `write_nbi` case (`LINK-001` .. `LINK-004`) the device program is a single
AIV kernel per submission batch that:

1. reads the `asc_comm_transfer[]` descriptor the host launcher enqueued;
2. for each entry, issues an asynchronous remote write (**`write_nbi`**) from the
   registered source buffer to the peer's registered destination buffer;
3. optionally sets the peer notify flag (`ASC_COMM_FLAG_NOTIFY`) **after** the write;
4. drains its own outstanding writes (`ASC_COMM_FLAG_DRAIN`) before the kernel exits,
   so kernel completion means "the writes retired", not "the writes were issued";
5. publishes the completion the host side waits for in `asc_comm_aiv_drain`.

The host never performs the copy. If the kernel is absent, the case is blocked.

## Required toolchain artifacts (check these first)

The device sources must be written against the headers **actually installed** on the
target machine. Before writing a single call, confirm each of the following exists and
record its path in the kernel header comment:

- [ ] `${ASCEND_HOME_PATH}/include` — Ascend C / AIV intrinsic headers for the
      installed toolkit version;
- [ ] the HCOMM device-side header that declares the remote-write and drain
      primitives for the AIV engine (the same header set the HCOMM bridge links
      against on the host);
- [ ] the Ascend C compiler driver (`ccec` or the CMake toolchain file the toolkit
      ships) and its version;
- [ ] the SoC identification of the target board (`npu-smi info`), because the UB
      write primitives differ between 950/A3 generations.

**No API signature is written down in this repository.** Guessing a vendor signature
is how an unverifiable kernel gets committed; the checklist above is the input to
writing `write_nbi/kernel.cpp`, not a description of it.

## Why `aiv/write_nbi/kernel.cpp` is not committed yet

Writing it without the headers would mean inventing the remote-write and drain
signatures. An invented kernel still compiles nowhere, still never runs, and would
turn a clearly reported blocker into a plausible-looking file that a reader may
believe. The instruction this project is built under is explicit: an uncompiled,
unexecuted kernel must be reported as blocked, not described as implemented.

The build system is therefore complete and the device source is the single missing
artifact:

```
kernels/asc_comm/aiv/write_nbi/kernel.cpp   <- to be written on the target machine
```

`CMakeLists.txt` picks it up automatically as soon as it exists
(`ASC_COMM_BUILD_AIV` is defined and the file is added to the library), and
`build.sh --strict` fails while it is missing.

## msprof evidence required

The kernel must be identifiable in a profiler record. Choose a stable, unique name and
put it in the kernel's `extern "C"` symbol; the evidence collector looks it up:

```
kernel_name_pattern:  asc_comm_aiv_write_nbi*
```

`baselines/l0/verification/evidence.py` accepts a `msprof_kernel` claim only when the
name in the record matches an executed kernel, so a case cannot claim kernel evidence
without one.

## Completion and the anti-early-return test

`wait()` must not return before the peer can read the data. Verify with the delayed
write test in `docs/l0_data_plane.md`:

1. rank 0 submits a remote write to rank 1;
2. rank 1 spin-reads the destination **without** synchronising;
3. rank 0 must not report completion until rank 1's read matches the pattern;
4. a failure here means the drain is a fence, and the case must be downgraded to
   `fenced` in `baselines/l0/transport/completion.py`.

## Performance metrics to collect

* kernel duration (msprof), compared against the host-side submission time;
* outstanding requests actually in flight (`outstanding` parameter);
* per-peer fan-out bytes, so `bandwidth` uses measured wire bytes rather than a
  model;
* AIV core utilisation, to show the copy actually ran on AIV.
