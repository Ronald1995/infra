# hierarchical_alltoallv_mindspeed

原生入口为 `MoELayerOverlapAll2All.apply`，使用 `preprocess_tp_extend_ep` 和 MindSpeed 的分层 token permutation。有明确的外部 AllGather/ReduceScatter 组和节点内 AllToAllV 组。 [MindSpeed 固定实现](https://gitcode.com/Ascend/MindSpeed/blob/3f2fb3701ea26507b739ee4c8ae5f04aea7fa556/mindspeed/core/transformer/moe/moe_layer_overlap_all2all.py) · [Megatron core_r0.8.0](https://github.com/NVIDIA/Megatron-LM/tree/baf94af3c667248865f23df73b9fb8e2395e6fd0)

- L3 共享固定路由。L4 可由统一 Runner 加入 Router。原生前反向执行通信及 GMM，适配层转换分片权重并回传输入、路由权重与本地专家参数梯度。使用限幅 SwiGLU，无共享专家。
- BF16 `[T,H]` 和等长 `[T,K]`。全局专家数能被物理 EP rank 数整除。显式设置 `hierarchical_inner_size` 和 `server_ids`，运行时核对单节点内层与实际进程组顺序。单节点环境不能启动此路径。
- 需要指定提交的 MindSpeed、Megatron、torch_npu 和可用的 MindSpeed GMM 扩展。当前没有目标集群执行证据。
- 通算边界见 [测试方法](../../../03_benchmark/ep_benchmark_methodology.md)。有效专家前向 FLOPs 为 `6 × routes × H × I`。 [结果归档](../../../05_results/raw/README.md)

```sh
python -m operators.hierarchical_alltoallv_mindspeed.benchmark \
  --config operators/hierarchical_alltoallv_mindspeed/config.yaml --cases MOE-003 \
  --nnodes 2 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29520 --output /shared/ep-runs \
  --run-id hierarchical-ep16 --session launch-hierarchical-ep16
```

## Benchmark contract

- **Cases it can run** (capability-level): none.
- **Cases it cannot run**: L1 独立通信 COMM-001, COMM-002, COMM-003, COMM-004, COMM-005, COMM-006, COMM-007, COMM-008, COMM-009, COMM-010, COMM-011, COMM-012; L2 串行组合 FUSE-001, FUSE-003; L2 原生融合 FUSE-002, FUSE-004, FUSE-005, FUSE-006, FUSE-007, FUSE-008, FUSE-009; L3 完整 MoE MOE-001, MOE-002, MOE-003, MOE-004, MOE-005, MOE-006, MOE-007, MOE-008, MOE-009, LAYER-007; L4 路由器/规划/去重 LAYER-001, LAYER-002, LAYER-003, LAYER-004, LAYER-005, LAYER-006, LAYER-008, LAYER-009.
- **Why not**: L3 replay needs a real_trace_replay workload; this configuration does not select one; deduplicated dispatch has no dedup path; the None control for this suite is COMM-001 with dedup.mode=none; native hierarchy requires at least two outer ranks, each with at least two inner ranks; the planner is disabled in this configuration; the load_balance suite measures this case with load_balance.enabled=true; this case measures rank dedup; the plan selects dedup.mode=none; this case measures server dedup; the plan selects dedup.mode=none.

- **Correctness contract**: `validate_moe` with the router gate for L4, so the router weight gradient is checked as well as the input and local expert weight gradients.
- **Timing contract**: `moe_case`; forward and, for a training case, backward run inside one timed call; the sample closes after `ctx.synchronize()`.
- **Registry constraints**: `hierarchical=true` requires `distributed.hierarchical_inner_size >= 2` with at least two outer ranks per node **and** an explicit `topology.server_ids`; otherwise `common/registry.py::check_constraints` reports `unsupported` ("native hierarchy requires ..."). Also `dtypes=[bfloat16]`, `equal_tokens`, `contiguous_placement`.
- **Source**: MindSpeed at commit `3f2fb3701ea26507b739ee4c8ae5f04aea7fa556`, Megatron `baf94af3c667248865f23df73b9fb8e2395e6fd0`; not ascendep-vendored. No target-cluster execution evidence exists.
