# mc2_dispatch_cann_v1

Native API: `npu_moe_distribute_dispatch / npu_moe_distribute_combine`. Source commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d`. [Source](https://github.com/Ascend/op-plugin)

- **Adapter** · `integrated`. V1 public bindings. The opaque reverse route stays in the native handle; correctness checks per-expert token multisets and weighted combine.
- **Layout** · Source tokens [T,H], routes [T,K] and rank-local weights. Native packed layouts follow the linked source. Constraints and tested semantics are recorded in the registry.
- **Benchmark** · L1/L2/L3. `dispatch`, `combine`, `moe_forward`. [Capabilities](../registry.yaml)
- **计时和统计** · 必需布局转换和通信计入相应用例。有效专家前向 FLOPs 为 `6 × routes × H × I`，完整反向为其两倍。通信模型按实际 trace 统计。 [方法与计时边界](../../../03_benchmark/ep_benchmark_methodology.md)
- **Dependencies** · Install the linked native components and a compatible torch_npu. Device execution and profiling require an Ascend host.
- **归档** · [测试矩阵](../README.md) · [EP8 历史报告](../../../05_results/ep_benchmark_ep8.md) · [结果索引](../../../05_results/raw/README.md)

```sh
python -m operators.mc2_dispatch_cann_v1.benchmark --config operators/mc2_dispatch_cann_v1/config.yaml
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-001, COMM-003, COMM-004, COMM-007, COMM-008, COMM-010; L2 FUSE-001, FUSE-003; L3 LAYER-007, MOE-001, MOE-005, MOE-006, MOE-008.
- **Cases it cannot run**: COMM-002/005/006/009/011/012 (`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`/`planned_dispatch`); FUSE-002/004/005/006/007/008/009; LAYER-001..006, LAYER-008, LAYER-009; MOE-002/003/004 (`moe_backward`), MOE-007 (`concurrent`), MOE-009 (`dynamic_shapes`). They stay in the manifest as `unsupported`.
- **Correctness contract**: `dispatch` returns an opaque reverse-route handle, so `validate_dispatch` checks per-expert token multisets and expert boundaries rather than row order; combine is checked with the weighted reference.
- **Timing contract**: L1 handlers submit and wait inside the timed block and record `local_complete`; the V1 group is created in `initialize` (`MC2V1Backend`), outside the timer.
- **Registry constraints**: `ep_sizes=[2,4,8,16,32,64,128,144,256,288]`, `dtypes=[bfloat16,float16]`, `hidden_size=7168`, `max_tokens=512`, `equal_tokens`, `contiguous_placement`.
- **Source**: op-plugin commit `c83e36b808c4e053c9245f95c7bc45cfaae0000d` (`operators/cann_special.py::MC2V1Backend`); not ascendep-vendored.
