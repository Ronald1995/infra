# 昇腾 EP Benchmark

以 [算子调研表](../../ep.md) 和 [测试方法](../03_benchmark/ep_benchmark_methodology.md) 定义范围。36 个算子的独立入口、已接入层次和限制见 [Capability Matrix](operators/README.md)。[SKILL.md](SKILL.md) 是 Agent 运行入口。

## 统一测试矩阵（Benchmark Plan）

一份 `configs/benchmark_plan.yaml` 生成 `EP Size × Suite × Case × Operator × Parameter Variant`
完整矩阵，默认覆盖 EP8/EP64/EP128 与 L0～L4 全部 Catalog。设计细节见
[测试矩阵设计](docs/benchmark_matrix.md)。

```sh
# 完整计划：只生成 Manifest，不执行
python run_benchmark.py plan --plan configs/benchmark_plan.yaml

# 完整执行：EP8、EP64、EP128 顺序执行，逐 Case 继续，失败不中断其他 EP Size
python run_benchmark.py run  --plan configs/benchmark_plan.yaml

# 能力视图 / 缩小范围
python run_benchmark.py capabilities --plan configs/benchmark_plan.yaml
python run_benchmark.py run --plan configs/benchmark_plan.yaml --ep-size 64
python run_benchmark.py run --plan configs/benchmark_plan.yaml --suite overlap
python run_benchmark.py run --plan configs/benchmark_plan.yaml --cases COMM-001,COMM-003
python run_benchmark.py run --plan configs/benchmark_plan.yaml --operators deepep_ascend,moonep

# 无 NPU 的 Plan → Manifest → 执行 → 覆盖率报告校验
python run_benchmark.py run --plan configs/benchmark_plan_smoke_cpu.yaml \
  --cases COMM-001,COMM-004,COMM-005,MOE-003,LAYER-002 --operators alltoall_hccl
```

分层规则：

- `configs/base_benchmark.yaml`：全局共享的测试定义（层级、消息大小候选池、L0 sweep、验证策略、固定计算 Shape）。
- `configs/ep{8,64,128}_bf16.yaml`：只描述 EP 环境（EP/TP、模型、专家数、拓扑来源），**不再决定 Case 范围**。
- `configs/{suite}.yaml`：只描述场景（workload、`benchmark` 维度、operators）；plan 组合时不会覆盖 EP 环境字段。
- `common/variants.py`：Case-specific 参数变体（不是全维度笛卡尔积）。

每个 task 保留 `config_source`、`config_fingerprint`、`ep_size`、`suite_ids`，不支持的组合保留
`unsupported / dependency_missing / not_implemented / not_applicable` 状态与原因。
`HW-007/HW-008/HW-009/HW-012` 是 `LINK-012/LINK-013/MEM-001/OVERLAP-001` 的 legacy alias，默认
不重复生成 task，但在 `manifest.matrix.case_index` 与覆盖率报告中显式记录。

执行后生成 `reports/coverage_matrix.{md,csv,json}`、`case_operator_coverage.csv`、
`operator_coverage.md`、`ep_size_comparison.md` 和 `l0_scaleout.md`；`run` 命令也会写入覆盖率报告。

## L0 硬件基线

`baselines/l0` 之外的 L0 入口在 [L0 README](baselines/l0/README.md)。L0 分六类：本地内存、原生通信机制、通信执行资源、集合通信、计算、通信与计算并发。每条结果区分 `requested` 与 `observed`，并给出 `verified` / `fallback` / `unverified` / `unsupported` 状态；只有 `verified` 结果可以跨通信机制比较。

L0 有独立 Suite 和配置，按 Case category 过滤，而不是靠 YAML 文件名推断：

| Suite | 配置 | Case |
|---|---|---|
| `hardware` | `configs/hardware.yaml` | HW-001～006、HW-010～011 |
| `memory` | `configs/memory.yaml` | MEM-001～004 |
| `transport` | `configs/transport.yaml` | LINK-001～013 |
| `collectives` | `configs/collectives.yaml` | EXEC-001～006 |
| `overlap` | `configs/overlap.yaml` | OVERLAP-001～009 + FUSE-001～009 + MOE-007 |


```sh
python run_benchmark.py capabilities --config configs/hardware.yaml
python run_benchmark.py run --config configs/transport.yaml --ep-size 8 --cases LINK-001,LINK-012,LINK-013
python run_benchmark.py run --config configs/collectives.yaml --ep-size 8 --cases EXEC-001
python run_benchmark.py run --config configs/overlap.yaml --ep-size 8 --cases OVERLAP-005
```

HCOMM 原生桥、SHMEM UDMA kernel、本地内存 kernel 和 oam-tools `hccl_test` 都是可选依赖：缺失时对应用例返回 `dependency_missing`，不影响其他 L0 及 L1–L4 测试。
## 安装和计划

在本目录安装框架。按目标 CANN 版本准备相容的 PyTorch 和 torch_npu。CPU 测试需要支持 Gloo 的 PyTorch。

```sh
python -m pip install -e '.[test,charts]'
python run_benchmark.py doctor
python run_benchmark.py list --capabilities
python run_benchmark.py plan --config configs/ep8_bf16.yaml --operators all --levels L0,L1,L2,L3,L4
```

由 ascendep 提供原生路径的算子读取固定源码 checkout。按上游模块 README 安装原生依赖。不同原生扩展可能提供同名 Python 包，可通过 `launch.operators.<id>.python` 和 `launch.operators.<id>.env` 使用独立环境。框架直接调用该 checkout 的源文件，仓库中没有复制上游实现。使用时遵循第三方源码和二进制的适用许可证。

```sh
git clone https://github.com/penn513/ascendep.git /path/to/ascendep
git -C /path/to/ascendep checkout 24ff84d3ab8c01e1e2071802465173e624e1f89d
export EP_ASCENDEP_ROOT=/path/to/ascendep
```

## 执行

```sh
# CPU 双 rank 校验
python run_benchmark.py pipeline --config configs/smoke_cpu.yaml \
  --cases COMM-001,COMM-004,COMM-005,MOE-003,LAYER-002 --run-id local-cpu

# 单卡 GMM/SwiGLU、双卡通信和 EP8 前反向
python run_benchmark.py baseline --config configs/smoke_npu.yaml --ep-size 1 \
  --cases HW-010,HW-011
python run_benchmark.py run --config configs/smoke_npu.yaml --ep-size 2 \
  --operators alltoall_hccl --cases COMM-001,COMM-003
python run_benchmark.py run --config configs/smoke_npu.yaml --ep-size 8 \
  --operators alltoall_hccl --cases MOE-003

# 单算子和公共基线
python -m operators.deepep_ascend.benchmark --config operators/deepep_ascend/config.yaml --levels L3
python run_benchmark.py baseline --config configs/hardware.yaml --levels L0
python run_benchmark.py baseline --config configs/ep8_bf16.yaml --levels L1,L2
# 批量运行、恢复及产物
python run_benchmark.py run --config configs/ep8_bf16.yaml --operators all
python run_benchmark.py resume --run-id RUN_ID
python run_benchmark.py profile --run-id RUN_ID --cases COMM-001,MOE-003
python run_benchmark.py analyze --run-id RUN_ID
python run_benchmark.py report --run-id RUN_ID
```

`--list-backends` 等价于 `list`，旧的 `--config PATH` 等价于 `run --config PATH`。单算子入口（`python -m operators.<id>.benchmark`）默认 L1–L4，因为 L0 是公共硬件基线；`--config` 未显式指定 `--levels` 时使用配置里的 `benchmark.levels`（EP 环境配置默认 L0–L4，`--plan` 始终使用 plan 的 levels）。未按层级或 Case 限定的 NPU `pipeline` 先运行公共基线，再运行算子用例。启用 `profiling.enabled` 时，每个后端、层级和配置组合选取一个已通过的用例单独采集 Trace，然后生成分析和报告。也可用 `profile` 独立采集已通过的用例。

## 多机和扩展性

每个节点启动同一份配置，使用共享输出根、run ID、session 和主节点端口。一次启动一个 EP Size。弱扩展固定每卡 token 和专家数，强扩展固定全局数量。运行时记录逻辑 rank、设备可见性、主机和设备型号。物理域由 `topology.server_ids`、`topology.supernode_ids` 显式配置，当前专家和 Router 工作负载使用 TP1。

```sh
launchers/run_ep64.sh --config configs/weak_scaling.yaml --ep-size 64 \
  --nnodes 8 --nproc-per-node 8 --node-rank "$NODE_RANK" \
  --master-addr "$MASTER_ADDR" --master-port 29520 \
  --output /shared/ep-runs --run-id weak-64 --session launch-weak-64
```

## 结果及状态

- `common/` 负责确定性分片、路由、分布式执行、计时和结果归档。
- `benchmarks/` 和 `baselines/` 负责硬件、通信、融合、专家层和 Router 用例。`LAYER-007` 以 L3 trace replay 归档。
- `metrics/`、`profiling/`、`analysis/`、`reporting/` 负责带宽模型、分位数、重叠区间、可比性与报告。
- `schemas/` 提供版本化配置及结果。基线只有在配置、环境指纹均匹配时才能复用。NPU 缓存当前限单节点。
- `--output` 可以改变归档根。默认见 [原始结果索引](../05_results/raw/README.md)。

`not_implemented` 表示对应能力待接入。历史实测和新运行分开保存。设备计数器缺失时保留空值。各算子的适用版本、形状和用例边界见 [算子接入矩阵](operators/README.md)。

专项入口包括 `configs/load_balance.yaml`、`configs/dedup.yaml` 和 `configs/overlap.yaml`。MoonEP 的训练用例对照固定归属 HCCL，并单独记录原生 fresh/cached plan 或 rank/server dedup。去重前后跨服务器字节数是对原始路由归属的 Dispatch 模型估算。报告中保留该计量边界。并发用例 MOE-007 在两条 NPU Stream 和两个 HCCL 通信组上提交两次独立专家前向。

分开执行的扩展点可用 `report --run-id weak-128 --compare-run weak-8 --compare-run weak-64` 合并分析。NPU Smoke 可通过设置 `EP_RUN_NPU_TESTS=1` 后运行 `python -m pytest -m npu` 执行。

开发环境执行 `python -m pytest -q` 验证。Windows CPU wheel 缺少 libuv 时使用 Gloo 文件 rendezvous。系统临时路径过长时，使用 `--basetemp` 指定短测试目录。
