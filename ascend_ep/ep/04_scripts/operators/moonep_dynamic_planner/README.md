# moonep_dynamic_planner

使用 MoonEP `dev_npu` 的原生规划分发、权重预取、逆向通信与冗余梯度归并。固定参考提交 `b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f`。 [源码](https://gitcode.com/Ascend/MoonEP/tree/dev_npu)

- L1 `COMM-012` 测量新规划和分发。
- L4 `LAYER-003` 测量完整训练周期，分别与合法缓存 plan 和固定归属 HCCL 训练对照比较。缓存保留冗余专家。MoonEP 固定提交的 Buffer 要求正数 `B`，固定归属对照采用公共 HCCL 实现。
- `LAYER-004` 在完成完整训练周期的同时，用设备事件记录权重预取和梯度归并。规划与分发连在同一 API 内，保留组合时间。
- `LAYER-008` 默认在每个样本内执行 4 个 Router/Top-K 前反向周期。配置通过轮换专家的 logit 偏置改变热点，每次重新规划并刷新冗余权重。`benchmark.planner_steps` 可配置。
- Workload 使用 BF16 `[T,H]`、FP32 原生路由标量、固定等长 `[T,K]` 和分片专家权重。对称内存只为本地专家和冗余槽位分配物理数据。原生分配要求本地专家数等于冗余槽位数。
- 安装固定提交的 MoonEP npu_patch 和 SHMEM。默认 UDMA 数据面、MTE 规划，跨机设置 `benchmark.shmem_rendezvous`。CPU 测试覆盖梯度衔接和槽位映射，设备性能尚未采集。
- [注册表](../registry.yaml) · [结果归档](../../../05_results/raw/README.md)

```sh
python run_benchmark.py run --config configs/load_balance.yaml \
  --cases COMM-012,LAYER-003,LAYER-004,LAYER-008
```

## Benchmark contract

- **Cases it can run** (capability-level): L1 COMM-010, COMM-012; L4 LAYER-003, LAYER-004, LAYER-008.
- **Cases it cannot run**: all L1 dispatch/combine cases except COMM-010/012 (`dispatch`/`combine`/`cached_dispatch`/`async_*`/`quantized_dispatch`/`combine_add_rmsnorm`); FUSE-001..009; L3 `moe_forward`/`moe_backward` cases (no `moe_forward`/`moe_backward`); LAYER-001/002/009 (`router_layer`), LAYER-005/006 (`rank`/`server_dedup`). They stay in the manifest as `unsupported`.
- **Correctness contract**: the planner cases run `operators/moonep_dynamic_planner/training.py::run_training_step` and are checked by `validate_moe(use_router=True)` plus planner-specific fields (`planned_receive_rows`, `planned_rank_max_mean`, `planned_expert_copies`, `planned_redundant_weight_bytes`).
- **Timing contract**: `fresh_plan` runs the planner every step; `cached_plan` plans once outside the timer and reuses it (`benchmarks/cases/l4_router_planner.py::_planned`). LAYER-008 executes `benchmark.planner_steps` (default 4) router/top-k cycles per sample. `stage_ns` feeds `metrics/summary.py::stage_latency`.
- **Registry constraints**: `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`, `min_tokens=1`; no `ep_sizes` pin. `common/scheduler.py` marks the fixed-owner variant of LAYER-003 separately (owner `alltoall_hccl`).
- **Source**: MoonEP `dev_npu` at commit `b4127baa1bec0bcccd49b48c2dc2a3be70da9a0f`; not ascendep-vendored, and no device performance has been collected.
