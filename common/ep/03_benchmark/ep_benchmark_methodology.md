# 昇腾 EP 算子统一 Benchmark 方案设计

建议将这套 Benchmark 的核心目标定义为：

> 在统一硬件、模型参数、Token 输入、路由分布和计算语义的条件下，对不同 EP 实现进行分层性能测试，量化通信效率、专家计算效率、通算融合收益、大规模扩展能力、负载均衡收益及资源消耗，并最终定位不同算子的性能瓶颈。

目前 [ascendep EP8 Benchmark](https://github.com/penn513/ascendep/blob/main/reports/ep8-profile/benchmark.md?utm_source=chatgpt.com) 已经提供了比较好的完整专家子层前反向性能基线，但对于后续 EP64/128 的性能选型，仅使用前反向耗时和 tokens/s 是不够的。

例如 MegaMoE 比 DeepEP 快，可能是因为 GMM 效率更高、通信计算重叠更充分，也可能是因为减少了 Token 重排和 Host 调度开销。需要通过分层测试区分这些因素。

## 一、Benchmark 总体设计

### 1.1 测试目标

| 评价维度 | 需要回答的问题                  | 核心指标                 |
| ---- | ------------------------ | -------------------- |
| 通信性能 | Dispatch/Combine 通信能力如何？ | Latency、GB/s         |
| 硬件效率 | 是否充分利用昇腾通信硬件？            | 带宽利用率、计算利用率          |
| 计算性能 | 专家 GMM 实现是否高效？           | TFLOPS、GMM 耗时        |
| 通算融合 | 通信与计算是否有效重叠？             | 融合加速比、通信隐藏率          |
| 整体性能 | 完整 EP 专家子层执行效率如何？        | 前反向耗时、tokens/s       |
| 扩展能力 | EP8 扩展到 EP128 后性能如何变化？   | Scaling Efficiency   |
| 负载均衡 | 不均衡路由会造成多大性能下降？          | MaxVio、Rank Max/Mean |
| 去重能力 | Rank/Server 去重能减少多少通信？   | 流量下降比例、性能收益          |
| 资源消耗 | 是否消耗过多计算核与显存？            | AIV/AIC 占用、HBM       |
| 稳定性  | 是否存在长尾、同步等待和慢卡？          | P95/P99、Rank 耗时分布    |

### 1.2 分层测试架构

| 层次 | 测试对象                         | 测试目的              |
| -- | ---------------------------- | ----------------- |
| L0 | HCCL、SHMEM、P2P、GMM 等独立基线     | 建立硬件和计算性能参考上限     |
| L1 | Dispatch、Combine             | 评价通信算子本身          |
| L2 | Dispatch + GMM、GMM + Combine | 评价单段通算融合能力        |
| L3 | 完整 Expert FFN 前反向            | 评价完整 EP 后端业务性能    |
| L4 | Router、Top-K、动态规划及专家前反向 | 评价真实 MoE 训练层性能 |

所有后端只在相同测试层次、相同计算语义下进行横向排名。

对于 MegaMoE 等无法独立拆分 Dispatch/Combine 的整层融合算子，L1 标记为 N/A，不应通过人为拆分 Kernel 或估算剩余时间获得所谓纯通信性能。

## 二、统一测试用例构造

测试用例的构造是整个 Benchmark 最重要的部分。如果不同 EP 实现使用不同路由分布、不同 GMM 计算量，即使模型参数相同，最终数据也不能公平比较。

### 2.1 基础参数定义

本方案覆盖 MoE 训练。测试输入规模按每卡源 Token 数定义。

| 参数 | 符号 | 建议测试值 |
| --- | --- | --- |
| EP Size | P | 8、16、32、64、128 |
| 每卡源 Token 数 | T | Dispatch 前本 Rank 的 Token 数。基础测点为 512、1024、2048、4096、8192，大输入测点按训练负载与后端容量向上扩展 |
| Hidden Size | H | 4096、7168、8192 |
| Expert Intermediate Size | I | 1024、2048、3072 |
| 全局专家数量 | E | 24、128、256、384 |
| Top-K | K | 采用各场景模型的实际路由 Top-K，同一对照组固定 K |
| 每卡专家数量 | E/P | 1、2、3、4、8 |
| Dispatch dtype | — | BF16、FP8 等 |
| Combine dtype | — | BF16、其他受支持类型 |
| Tensor Parallel | TP | EP8 基线固定为 1，其余测点记录场景模型的实际配置 |
| 路由分布 | — | 均衡、随机、热点、真实训练 Trace |
| 去重策略 | — | None、Rank、Server |
| 动态专家规划 | — | Off、On |
| 执行模式 | — | 训练前向、训练反向、完整前反向 |

记录各 Rank 在 Dispatch 前的实际输入 Token 数。存在补齐或复制时，同时记录用于吞吐统计的有效源 Token 数。

原生算子有单次调用上限时，按训练路径分块。保留完整测点的每卡源 Token 数，另记块大小、次数与尾块，并将全部分块纳入计时和显存统计。

上述参数不做全量笛卡尔积组合，而是构造具有明确测试目的的专项测试集合。

### 2.2 统一输入数据

所有 Backend 使用同一份 Workload 和 Routing Trace。

```text
hidden_states: [T, H]
topk_ids:      [T, K]
topk_weights:  [T, K]
w_gate:        [local_experts, I, H]
w_up:          [local_experts, I, H]
w_down:        [local_experts, H, I]
grad_output:   [T, H]
```

支持融合 Gate/Up 权重的 Backend 可以在初始化阶段转换权重布局，但不能改变数学计算语义。

需要保证相同专家 ID 对应相同权重、相同数学计算、相同激活函数及限幅策略、相同输出梯度。每次测试保存随机种子及 Routing Trace Hash。

对于 L1、L2、L3，提前生成 Router 的 Top-K 结果，不计入 Router Linear 和 Top-K 选择耗时；但后端自身必需的路由转换和准备操作需要单独统计，并在完整 API 计时中纳入。

只有 L4 才包含 Router 的实际计算。

### 2.3 路由分布构造

不同路由分布不仅影响专家计算量，也会直接影响网络通信量、目的 Rank 分布和通信拥塞。

#### Case A：完全均衡路由

保证全局每个专家收到相同或尽可能接近的 Token 数，同时控制各源 Rank 到目标 Rank 的流量均衡。

适用于测量理想条件下的通信带宽、GMM 性能和通算融合上限。

#### Case B：随机均衡路由

从全部专家中随机、不重复地选择 K 个专家。

整体负载接近均衡，但保留自然的统计波动，比完全均衡路由更接近普通业务。

#### Case C：热点专家路由

人为控制少数专家接收更多 Token。

##### 专家负载不均衡测试用例

热点强度示例为 1×（均衡）、1.5×、2×、4×、8×。

| 指标 | 示例值 |
| --- | ---: |
| 平均负载 | 100 |
| 热点专家负载 | 200 |
| 最大 / 平均 | 2.0× |

归一化负载分布示意，保持总路由工作量不变。真实 Trace 生成时需满足整数计数、Top-K 不重复等约束。

建议分别构造两类不均衡：

| 类型             | 构造方式                 | 测试目的             |
| -------------- | -------------------- | ---------------- |
| Expert Hotspot | 少数专家负载过高，但分布于不同 Rank | 测试专家 GMM 负载差异    |
| Rank Hotspot   | 热点专家集中部署在少数 Rank     | 测试通信拥塞和慢 Rank 效应 |

这样才能区分专家计算负载不均衡和网络通信负载不均衡。

#### Case D：通信局部性路由

保持专家负载相同，改变 Token 到目标 Rank 的位置关系。

- Locality High
- Uniform AllToAll
- Cross-Server Heavy
- Destination Hotspot

流量矩阵以源 Rank 和目标 Rank 为两个维度，每格的流量由低到高。最后一种场景同时引入目标 Rank 热点。

这种测试对于 EP64/128 非常重要。即使专家负载相同，跨 Server 的通信比例不同，实际通信性能也可能完全不同。

#### Case E：去重路由

控制一个 Token 的 K 个专家在多少个不同 Rank、Server 上分布。

按场景模型固定 K，根据专家放置构造不同的去重机会。

| 场景 | 不同专家数 | 目标 Rank 数 | 目标 Server 数 |
| --- | --- | --- | --- |
| 无 Rank 去重机会 | K | K（拓扑允许时） | 视拓扑而定 |
| Rank 去重 | K | 少于 K | 视拓扑而定 |
| Rank + Server 去重 | K | 少于 K | 少于目标 Rank 数 |
| 高局部性 | K | 按域内专家放置确定 | 1 |

此类 Trace 必须根据每卡专家数量和实际拓扑构造，不能生成数学上无法满足的路由。

用于评价 MoonEP Rank/Server Dedup、DeepEP 去重和分层转发等优化。

#### Case F：真实模型路由

从实际 MoE 模型的训练过程中采集路由 Trace，保留每层专家分布、Token 数量、Rank 局部性以及动态变化特征。

合成路由用于性能归因，真实路由用于验证优化是否具有实际业务价值。

## 三、各层 Benchmark 测试用例详细设计

### 3.1 L0：硬件通信与计算基线

L0 不直接评价某个 EP 库，而是建立对应硬件、拓扑和通信机制下的性能参考基线。

| 用例 ID  | 测试对象                      | 测试目的                 |
| ------ | ------------------------- | -------------------- |
| HW-001 | P2P 单向通信                  | 测量两个 Rank 间的单向带宽     |
| HW-002 | P2P 双向通信                  | 测量双向并发通信能力           |
| HW-003 | Multi-Peer P2P            | 测量多目的 Rank 并发通信能力    |
| HW-004 | HCCL AllToAll             | 均匀集合通信性能             |
| HW-005 | HCCL AllToAllV            | 不等长集合通信性能            |
| HW-006 | AllGather / ReduceScatter | AllGather 类 EP 的通信基线 |
| HW-007 | SHMEM MTE                 | MTE 数据搬运能力           |
| HW-008 | SHMEM URMA/UDMA           | 对应通信引擎的数据搬运能力        |
| HW-009 | HBM Copy                  | 本地显存带宽               |
| HW-010 | GroupedMatMul             | 独立专家计算基线             |
| HW-011 | SwiGLU                    | 激活计算性能               |
| HW-012 | GMM + Communication       | 通信与计算并发时的资源竞争        |

#### 通信测试数据量

建议按照指数增长方式覆盖：

```text
4 KiB
16 KiB
64 KiB
256 KiB
1 MiB
4 MiB
16 MiB
64 MiB
256 MiB
512 MiB
1 GiB
```

具体上限根据可用内存和接口限制决定。

理论趋势为小消息通常受固定延迟限制，大消息逐渐进入带宽主导区间。

L0 最终需要输出以下基准数据：同一通信域下的单向和双向带宽、跨 Server 带宽、多 Rank 并发带宽，以及对应数据类型和实际 Shape 的 GMM TFLOPS。

后续 EP 算子的性能不仅与芯片标称峰值比较，也与这些可实际达到的硬件基线比较。

### 3.2 L1：纯 EP 通信测试

这一层是评价 DeepEP、MoonEP、CANN Dispatch/Combine 等通信算子的主要依据。

#### 测试边界

`Dispatch → 预构造专家输出 → Combine`

Combine 使用预构造的专家输出，不需要实际执行 GMM。

#### 用例定义

| 用例 ID    | 测试内容                     | 计时范围                            |
| -------- | ------------------------ | ------------------------------- |
| COMM-001 | Dispatch Full            | 路由准备、重排、通信、输出布局                 |
| COMM-002 | Dispatch Cached          | 复用有效 Routing Handle 后的数据通信及必要处理 |
| COMM-003 | Combine Full             | 输入专家输出到最终 Token 加权归并            |
| COMM-004 | Dispatch + Combine       | 完整通信闭环，不含 GMM                   |
| COMM-005 | Dispatch Async           | 异步提交、等待和结果可用时间                  |
| COMM-006 | Combine Async            | 异步提交、等待和结果可用时间                  |
| COMM-007 | Training Tail Dispatch | 训练分块尾部的通信延迟 |
| COMM-008 | High-Throughput Dispatch | 大 Token 数通信带宽                   |
| COMM-009 | Quantized Dispatch       | 量化、通信及必要反量化                     |
| COMM-010 | Deduplicated Dispatch    | Rank/Server 去重通信                |

对于支持 Cached Handle 的算子，必须分别测试 Cached 和 Non-Cached 两种模式。否则一方包含路由准备，另一方不包含，会造成不公平的性能对比。

#### Dispatch 内部耗时拆分

逻辑阶段示意，实际实现可能交叠执行，不能简单累加所有阶段耗时。

建议尽可能采集：

| 指标                     | 含义                         |
| ---------------------- | -------------------------- |
| `layout_latency_us`    | 路由计数、目的 Rank 计算等           |
| `permute_latency_us`   | Token 打包和重排                |
| `issue_latency_us`     | Host 或设备侧提交通信任务的开销，分别标明计时域 |
| `transport_latency_us` | 通信数据面实际执行时间                |
| `drain_latency_us`     | 发出通信任务后等待传输完成的相关时间         |
| `epilogue_latency_us`  | 接收整理、同步或后处理时间              |
| `ready_latency_us`     | 从调用开始到输出真正可被下一算子使用的时间      |
| `api_latency_us`       | API 调用时间，标明同步或异步语义         |

对于无法通过公开接口或 Profiling 精确拆分的阶段，标记为 `unavailable`，不能随意将 Kernel 总时间当作 URMA 或 UDMA 传输时间。

### 3.3 L2：单段通算融合测试

主要用于比较以下实现：

- CANN AlltoAllvGroupedMatMul / GroupedMatMulAlltoAllv。
- FSDPTurbo MC2。
- lueelu EP_GMM_FUSED。
- 具有对应单段融合能力的其他后端。

- 非融合基线
- 分块通算融合
- Kernel 级通算融合

三种执行方式的概念示意，不代表实际耗时比例。

#### 测试用例

| 用例 ID    | 测试内容                   |
| -------- | ---------------------- |
| FUSE-001 | 独立 Dispatch + 独立 GMM1  |
| FUSE-002 | Dispatch-GMM1 原生融合算子   |
| FUSE-003 | 独立 GMM2 + 独立 Combine   |
| FUSE-004 | GMM2-Combine 原生融合算子    |
| FUSE-005 | 不同 Chunk Size 下的流水重叠   |
| FUSE-006 | 不同 Tile Size 下的通算融合    |
| FUSE-007 | 通信核数量变化对 GMM 的影响       |
| FUSE-008 | 量化通信与 GMM 融合           |
| FUSE-009 | 反向通信与 Dgrad/Wgrad 计算融合 |

所有融合测试至少需要三组基线：独立通信耗时、独立计算耗时、非融合串行执行耗时。

这样才能区分融合收益究竟来自通信隐藏、计算优化，还是中间数据搬运和算子启动开销的减少。

### 3.4 L3：完整专家子层测试

这一层与现有 ascendep Benchmark 最接近，是所有具备相同前反向能力的后端都应该参与的测试。

执行阶段包括 Forward 和 Backward。

| 用例 ID   | 测试内容                           | 主要评价指标                   |
| ------- | ------------------------------ | ------------------------ |
| MOE-001 | Forward Only                   | Forward Latency、tokens/s |
| MOE-002 | Backward Only                  | Backward Latency         |
| MOE-003 | Forward + Backward             | Training Throughput      |
| MOE-004 | Forward + Backward + Recompute | 重计算性能及显存收益               |
| MOE-005 | 不均衡路由                          | 负载敏感度                    |
| MOE-006 | 不同 dtype                       | 量化性能收益                   |
| MOE-007 | 多 Stream 并发                    | 并发及资源竞争                  |
| MOE-008 | 多层连续执行                         | Buffer 复用和持续吞吐           |
| MOE-009 | 动态 Token 数                     | 动态 Shape 适应能力            |

需要特别强调：训练反向测试必须验证输入梯度、两组专家线性层权重梯度以及路由权重梯度的计算语义一致。

如果某个实现没有计算 Wgrad，或者采用了不同的 Activation Recompute 策略，就不能直接将其与其他完整训练实现比较。

### 3.5 L4：完整业务 MoE Layer 测试

L4 需要引入真实的 Router，以及根据配置启用的动态专家规划、权重预取和梯度归并。

| 用例 ID     | 测试内容                             |
| --------- | -------------------------------- |
| LAYER-001 | Router + Top-K + Expert Forward  |
| LAYER-002 | 完整 MoE Layer Forward + Backward  |
| LAYER-003 | Dynamic Planner Off / On         |
| LAYER-004 | 冗余专家权重预取及梯度归并                    |
| LAYER-005 | Rank Dedup Off / On              |
| LAYER-006 | Server Dedup Off / On            |
| LAYER-007 | 真实模型 Routing Trace 回放            |
| LAYER-008 | 连续多个训练 Step 的动态负载均衡              |
| LAYER-009 | 大源 Token 数下的完整 MoE 训练层前反向 |

这层测试特别适合评估 MoonEP。仅仅测试关闭 Planner 的 `moonep` Dispatch/Combine，无法完整评价其动态冗余专家方案。

## 四、核心性能评价指标设计

我建议将指标划分为八个类别，其中通信带宽、通算融合收益和 EP 扩展效率是本次 Benchmark 最需要完善的部分。

### 4.1 延迟指标

对于所有 EP 算子，不能只记录平均耗时，因为集合通信及 MoE 执行经常被最慢的 Rank 限制。

假设第 $i$ 次迭代、第 $r$ 个 Rank 的耗时为 $t_{i,r}$，则每次迭代的整体耗时估计为：

$$
t_i=\max_r(t_{i,r})
$$

基于所有迭代的 Rank Max 计算 P50、P95、P99。

| 指标           | 含义           | 作用          |
| ------------ | ------------ | ----------- |
| Rank Min     | 最快 Rank 耗时   | 观察理想执行情况    |
| Rank Avg     | 所有 Rank 平均耗时 | 观察总体性能      |
| Rank Max     | 最慢 Rank 耗时   | 评价整个 EP 组耗时 |
| P50          | 中位数          | 主要性能排名      |
| P95          | 95% 样本不超过的耗时 | 长尾性能        |
| P99          | 99% 样本不超过的耗时 | 极端延迟        |
| P95/P50      | 长尾程度         | 稳定性         |
| Rank Max/Avg | Rank 耗时不均衡程度 | 慢卡分析        |
| CV           | 标准差 / 平均值    | 性能波动        |

Rank Max 是使用各 Rank 本地计时得到的整体耗时估计，不能直接将不同设备时钟的时间戳相减。必要时需要额外采集经过时钟对齐的全局 Trace。

此外，完整 Forward + Backward 耗时应该独立测量，而不是将 Forward P50 和 Backward P50 相加。

### 4.2 通信带宽指标

这是最容易产生错误结论的指标。

不同 EP 算子可能使用 Expert Expanded、Rank Dedup、Server Dedup、AllGather 或分层转发等不同通信方式。因此，需要区分三种通信带宽。

| 带宽类型               | 分子                   | 用途        |
| ------------------ | -------------------- | --------- |
| Logical Bandwidth  | 逻辑上需要处理的专家 Token 数据量 | 衡量业务通信效率  |
| Payload Bandwidth  | 实际跨指定通信域传输的有效数据量     | 衡量通信数据面效率 |
| Physical Bandwidth | 链路实际传输的字节数，含协议等开销    | 衡量硬件链路利用率 |

NCCL Tests 同样区分算法带宽与用于衡量硬件利用程度的总线带宽。但 EP 路由具有不等长通信及去重特征，不能简单套用普通 AllToAll 的固定带宽换算系数。

#### 逻辑通信量

设每卡 Token 数量为 T、隐藏维度为 H、Top-K 为 K、每元素字节数为 B。

每个 Rank 的专家展开逻辑数据量：

$$
V_{\mathrm{logical}}=T\times K\times H\times B
$$

其中包含本地专家路由和重复目的 Rank，因此不能直接作为物理网络流量。

#### Rank 去重后的通信量

如果同一个 Token 的多个专家都位于同一个 Rank，而且实现支持 Rank 去重，那么可以只发送一次 Hidden States。

未去重：发送 4 份 Hidden States

Rank 去重：只发送 2 份 Hidden States

本例 Hidden States 的跨 Rank 数据量减少 50%，但专家计算次数仍然为 4 次。

设一个 Token 的 Top-K 专家对应的不同远端 Rank 集合为 $R_t$，那么采用 Rank Dedup 时，源 Rank 发出的 Hidden States 有效数据量为：

$$
V_{\mathrm{rank}}= \sum_{t=1}^{T}|R_t|\times H\times B
$$

对于 Server Dedup，则需要根据实际拓扑和分层转发机制计算跨 Server 流量。Server 内的二次分发不能被忽略。

以上公式只计算 Hidden States，实际 Payload 还应记录索引、路由权重、量化 Scale、Padding、同步信息等附加数据，以及真实实现是否采用合并传输。

#### EP Dispatch 通信量估算示例

| 输入项 | 示例配置 |
| --- | --- |
| EP Size | 8 |
| 每卡 Token 数 | 4096 |
| Hidden Size | 7168 |
| Top-K | 6 |
| 通信数据类型 | BF16（另可估算 FP8） |
| 实测 Dispatch 耗时 | 按实测值填入，单位 ms |
| 去重后剩余远端流量比例 | 示例为 100%，刻度含 10%、50%、100% |

| 输出项 | 示例值 |
| --- | ---: |
| 每卡逻辑展开数据量 | 336.0 MiB |
| 未去重远端数据量 | 294.0 MiB |
| 去重后远端数据量 | 294.0 MiB |
| 节省通信数据量 | 0.0 MiB |
| 逻辑带宽 | 352.3 GB/s |
| 远端有效 Payload 带宽 | 308.3 GB/s |

假设专家在各 Rank 均匀部署且路由均匀；去重比例为假设输入。仅计算 Hidden States 的单向发送量，不含元数据、Padding、FP8 Scale、协议开销和多跳流量。正式测试必须使用实际 Trace 及通信计数。

此估算示例采用已有 EP8 基线参数。新训练测点按场景模型选取 Top-K。

一个非常重要的结论是：采用去重的算子可能 Payload Bandwidth 更低，但是业务性能更好。 因为它完成同样专家计算所需要传输的数据更少。

因此每个算子必须同时报告耗时、逻辑带宽、实际通信量以及 Payload Bandwidth。

### 4.3 硬件利用率

建议分别报告两类效率。

硬件理论带宽利用率：

$$
\eta_{\mathrm{peak}} =\frac{BW_{\mathrm{physical}}} {BW_{\mathrm{theoretical}}}
$$

实测基线效率：

$$
\eta_{\mathrm{baseline}} =\frac{BW_{\mathrm{payload,EP}}} {BW_{\mathrm{payload,L0}}}
$$

前提是分子和分母使用相同通信域、相同方向、相同字节统计口径。

| 指标              | 比较对象                 |
| --------------- | -------------------- |
| 单 Rank TX 带宽利用率 | Rank 可用单向发送带宽        |
| 单 Rank RX 带宽利用率 | Rank 可用单向接收带宽        |
| Server 内通信效率    | 对应 Server 内拓扑及基线     |
| 跨 Server 通信效率   | 实际跨 Server 链路或网络割集带宽 |
| HBM 带宽利用率       | 实测显存读写带宽             |
| AIV 利用率         | AIV 执行与资源占用          |
| AIC 利用率         | AIC 计算与资源占用          |

对于昇腾 950，不能直接将整个芯片 UB 聚合双向理论带宽作为所有 EP 算子的带宽分母。

应根据实际使用的端口、路由、通信引擎、通信方向和拓扑确定理论上限。如果暂时无法获得物理链路流量，则只报告逻辑带宽和 Payload 带宽，不给出虚假的硬件利用率。

### 4.4 专家计算性能指标

对于 BF16 SwiGLU MoE，专家 FFN 通常包含 Gate、Up、Down 三次矩阵乘法。

假设一个 Rank 实际处理的专家展开 Token 数量为 M，则前向理论有效 FLOPs 约为：

$$
F_{\mathrm{fwd}}=6MHI
$$

常规完整反向包含输入梯度和权重梯度计算：

$$
F_{\mathrm{bwd}}\approx12MHI
$$

所以完整前反向有效 FLOPs 约为：

$$
F_{\mathrm{train}}\approx18MHI
$$

这里不包含 SwiGLU、路由加权、额外重计算和 Padding 等操作。

| 指标                 | 含义                          |
| ------------------ | --------------------------- |
| Effective TFLOPS   | 基于实际有效专家 Token 的计算吞吐        |
| Executed TFLOPS    | 包含 Padding、重计算等实际执行的矩阵乘工作量  |
| GMM Efficiency     | 独立 GMM 性能相对对应 Shape 的 L0 基线 |
| Cube Efficiency    | 实测计算吞吐相对相同 dtype 的理论算力      |
| Padding Ratio      | 无效 Padding 计算比例             |
| Recompute Overhead | 重计算引入的额外计算开销                |

完整融合 Kernel 的 TFLOPS 应标记为端到端有效计算吞吐，而不是独立 GMM 的硬件计算效率。

这能够避免将 MegaMoE 的 GMM 计算性能和 DeepEP 的纯通信性能混为一谈。

### 4.5 通算融合指标

通算融合至少要从三个维度评价。

#### 指标 A：实际融合加速比

$$
S_{\mathrm{fuse}}= \frac{T_{\mathrm{serial}}} {T_{\mathrm{fused}}}
$$

其中非融合串行基线必须包含相同输入、输出、通信量和计算量。

#### 指标 B：串行执行时间节省比例

$$
R_{\mathrm{saving}}= \frac{T_{\mathrm{serial}}-T_{\mathrm{fused}}} {T_{\mathrm{serial}}}
$$

它衡量整体耗时收益，不等价于通信隐藏率。

#### 指标 C：可观测通信重叠率

$$
R_{\mathrm{overlap}}= \frac{T_{\mathrm{comm}\cap\mathrm{compute}}} {T_{\mathrm{comm}}}
$$

其中计算与通信区间应采用时间轴区间交集，而不是简单累加多个并发 Kernel 的时间。

#### 通算融合收益分析

| 指标 | 示例值 |
| --- | ---: |
| 独立通信耗时 | 6.0 ms |
| 独立计算耗时 | 10.0 ms |
| 融合 Kernel 耗时 | 12.0 ms |
| 理想化串行基线（通信 + 计算） | 16.0 ms |
| 融合加速比 | 1.33× |
| 总耗时节省比例 | 25.0% |

此处以独立通信与计算时间之和模拟串行基线。正式 Benchmark 必须实测串行基线；节省的时间不一定全部来自通算重叠。

对于 Domino、ChunkMoE 等多 Stream 重叠，MindStudio 的 Overlap Analysis 可以直接提供部分可见通信重叠数据。

对于 MegaMoE、CANN MC2 等 Kernel 内部融合，则可能需要内部时间戳、硬件计数器或者算子级插桩分析 AIV、AIC、通信引擎的流水。MindStudio 的可见未重叠通信指标是其识别出的通信区间减去计算通信重叠区间，不必然包括融合 Kernel 内部的全部通信活动。

因此，不能用 Profiler 显示的 0% 重叠率来评价 MegaMoE 没有通算重叠。

### 4.6 负载均衡指标

设专家 e 收到的 Token 数量为 $L_e$。

定义专家负载不均衡比例：

$$
R_{\mathrm{expert}}= \frac{\max_e L_e}{\operatorname{mean}_e L_e}
$$

对应的最大相对超载程度：

$$
V_{\max}=R_{\mathrm{expert}}-1
$$

建议同时记录 Expert Max/Mean 和这个明确定义的 MaxVio 风格指标。不同项目内部的 MaxVio 定义如果不同，应通过适配器转换或分别保留，不能直接混用。

| 指标                      | 含义                  |
| ----------------------- | ------------------- |
| Expert Max/Mean         | 专家负载不均衡程度           |
| Rank Max/Mean           | 各 Rank 专家计算工作量不均衡程度 |
| Rank TX Max/Mean        | 发送通信流量不均衡           |
| Rank RX Max/Mean        | 接收通信流量不均衡           |
| Server Traffic Max/Mean | Server 间流量不均衡       |
| Planner Latency         | 动态规划耗时              |
| Prefetch Latency        | 冗余专家权重预取耗时          |
| Reduce Grad Latency     | 冗余专家梯度归并开销          |
| Redundant Weight Bytes  | 额外权重显存消耗            |
| Net Speedup             | 包含规划及权重、梯度交换后的实际收益  |

对于 MoonEP 等动态负载均衡方案，最终收益必须基于包含 Planner、权重预取和梯度归并的完整执行周期。

如果规划每 N 步执行一次，则应该测试完整 N 步周期的平均收益和尾延迟，不能只测规划完成后的最快执行时间。

### 4.7 显存及硬件资源指标

EP 算子性能更高，也可能意味着其使用了更大的通信 Buffer 或占用了更多 AIV 核。

| 指标                         | 目的              |
| -------------------------- | --------------- |
| Communication Buffer Bytes | 通信缓冲区开销         |
| SHMEM/Symmetric Heap Bytes | 对称内存窗口开销        |
| Workspace Bytes            | 临时计算与通信空间       |
| Activation Bytes           | 训练激活保存开销        |
| Redundant Expert Bytes     | 冗余专家权重开销        |
| Peak HBM Allocated         | 框架实际分配峰值        |
| Peak Device HBM Used       | 设备实际显存使用量       |
| AIV / AIC Cores Used       | 通信及计算资源占用       |
| Device Idle Ratio          | 设备空闲比例          |
| Concurrent GMM Slowdown    | 通信并发对 GMM 性能的影响 |

尤其是 SHMEM 和第三方通信库直接分配的 Buffer，可能不在 PyTorch allocator 的统计范围中，所以需要同时采集框架分配和设备级显存数据。

### 4.8 EP 扩展性能指标

EP8 扩展到 EP64/128 必须至少区分两种扩展策略。

Weak Scaling：固定每卡 Token 和每卡专家数

增加硬件和总工作量，观察每卡负载不变时，通信及整体延迟是否恶化。

Strong Scaling：固定全局 Token 和全局专家数

保持同一模型和总工作量，通过更多 Rank 分摊专家与 Token。

Weak Scaling 效率：

$$
\eta_{\mathrm{weak}}(P)= \frac{\mathrm{TPS}(P)/\mathrm{TPS}(P_0)} {P/P_0}
$$

Strong Scaling 效率：

$$
\eta_{\mathrm{strong}}(P)= \frac{T(P_0)/T(P)} {P/P_0}
$$

#### Weak Scaling 扩展效率计算示例

假设各 Rank 固定 4096 个源 Token，以各规模的前反向耗时（ms）计算吞吐和扩展效率。

| 规模    | 吞吐 tokens/s | 扩展效率   |
| ----- | ----------- | ------ |
| EP8   | 1,092,267   | 100.0% |
| EP64  | 7,489,829   | 85.7%  |
| EP128 | 11,650,844  | 66.7%  |

默认耗时是假设值，并非实际 EP64/128 测试结果。

Weak Scaling 更适合观察 EP 通信库的大规模扩展能力；Strong Scaling 更适合观察固定模型在增加设备后实际能获得多少性能收益。

## 五、具体测试用例矩阵

不建议一次执行所有组合。应该将 Benchmark 划分为以下八个专项测试集合。

### Suite 01：EP8 基础性能测试

复用现有 ascendep Benchmark 配置。

| 参数              | 配置           |
| --------------- | ------------ |
| 硬件              | Ascend 950DT |
| EP              | 8            |
| Hidden Size     | 7168         |
| FFN Size        | 3072         |
| Experts         | 24           |
| Local Experts   | 3            |
| Top-K           | 6（复现已有 EP8 基线） |
| Tokens per Rank | 4096         |
| dtype           | BF16         |
| Routing         | 完全均衡         |
| Planner         | Off          |
| 测试层次            | L0、L1、L2、L3  |

目标是验证新 Benchmark 框架的正确性，与现有 22 个成功采样的 Backend 性能数据进行对照，并补齐纯通信和单段融合测试。

### Suite 02：大 Token 数 MoE 训练测试

按每卡源 Token 数扩大训练输入，候选测点为 8192、16384、32768、65536、131072。EP 从 8、64、128 中按场景模型和拓扑选择，结合算子容量和显存预算筛选测点。

| 参数 | 配置 |
| --- | --- |
| 模型参数 | Hidden Size、专家数和 Top-K 按场景模型配置 |
| dtype 和路由 | 使用场景模型的训练精度及训练路由 |
| 测试层次 | L3 完整专家子层前反向、L4 完整 MoE 训练层 |
| 主要指标 | 完整输入的前向、反向和前反向耗时，有效源 Token 吞吐、峰值显存及尾块开销 |

完整测点的计时包含所有分块和训练路径必需的同步。记录各 Rank 的实际输入、原生调用上限、块大小、次数及尾块。

### Suite 03：高吞吐通信测试

覆盖训练中源 Token 量较大的通信与通算融合路径。

| 参数              | 配置                          |
| --------------- | --------------------------- |
| EP              | 8、64、128                    |
| Hidden Size     | 7168                        |
| Experts         | 256                         |
| Top-K           | 按场景模型配置，同组固定 |
| 每 Rank 调用块大小（Token） | 256、512、1024、2048、4096、8192，按训练分块和接口能力选取 |
| dtype           | BF16、FP8                    |
| Routing         | 均衡、随机                       |
| 测试层次            | L1、L2、L3                    |

主要评价 Payload Bandwidth、实测带宽基线效率、GMM TFLOPS、完整吞吐及通信缓冲区消耗。

如果某个算子单次最大支持 512 Tokens，而测试输入为 4096 Tokens，需要明确记录其 Chunking 策略以及实际调用次数，并将全部分块执行时间纳入结果。

### Suite 04：EP 大规模扩展测试

采用两组独立配置。

| 参数         | Weak Scaling   | Strong Scaling |
| ---------- | -------------- | -------------- |
| EP         | 8、16、32、64、128 | 8、16、32、64、128 |
| 全局专家数      | 4 × EP         | 256            |
| 每卡专家数      | 固定 4           | 256 / EP       |
| 每卡 Token 数 | 固定 4096        | 32768 / EP     |
| 全局 Token 数 | 4096 × EP      | 固定 32768       |
| H / I      | 7168 / 2048    | 7168 / 2048    |
| Top-K      | 按场景模型配置，同组固定 | 按场景模型配置，同组固定 |
| dtype      | BF16           | BF16           |

注意 Weak Scaling 中全局专家数随 EP 变化，因此它是通信及计算扩展实验，不是同一个模型在不同设备数量下的性能实验。

这里采用独立于原 EP8 的大规模配置，是为了保证 EP128 时专家数能够合法、均匀地分配。

### Suite 05：通信拓扑及局部性测试

| 用例       | 路由条件                  | 核心目的          |
| -------- | --------------------- | ------------- |
| TOPO-001 | 优先本 Rank 专家           | 本地计算比例影响      |
| TOPO-002 | 优先同 Server 专家         | Server 内通信性能  |
| TOPO-003 | 均匀全局路由                | AllToAll 通信性能 |
| TOPO-004 | 优先跨 Server 专家         | 跨 Server 通信性能 |
| TOPO-005 | 目的 Rank 热点            | Rank 级拥塞      |
| TOPO-006 | 目的 Server 热点          | Server 级拥塞    |
| TOPO-007 | 相同 EP，不同 Rank Mapping | 物理组网映射敏感度     |

具体同 Server、跨 Server 的定义以实际机器拓扑为准，不假设所有设备均按照连续 8 Rank 组成通信域。

### Suite 06：动态负载均衡测试

| 参数              | 配置                         |
| --------------- | -------------------------- |
| EP              | 64、128                     |
| Tokens per Rank | 1024、4096                  |
| Expert Max/Mean | 1.0、1.25、1.5、2、4、8         |
| Planner         | Off、On                     |
| 冗余专家预算          | 0%、5%、10%、20%              |
| 路由变化            | 固定、缓慢变化、突变                 |
| Planner 周期      | 每步、每 10 步、每 50 步           |
| 测试模式            | Forward、Forward + Backward |

对于不具备动态冗余专家能力的 Backend，使用原生静态路由作为对照。

### Suite 07：Rank/Server 去重测试

| 用例        | 去重模式          | 对照方式               |
| --------- | ------------- | ------------------ |
| DEDUP-001 | None          | 原始专家展开通信           |
| DEDUP-002 | Rank          | 同一逻辑路由开启 Rank 去重   |
| DEDUP-003 | Server        | 同一逻辑路由开启 Server 去重 |
| DEDUP-004 | Rank + Server | 两级去重               |
| DEDUP-005 | 不同重复率         | 改变路由 Trace 中的去重机会  |
| DEDUP-006 | 真实路由          | 测量业务中的实际去重收益       |

对照时保持逻辑专家 ID、专家权重及输出语义不变，避免因为路由改变了 GMM 计算分布而影响结论。

### Suite 08：通算融合专项测试

| 参数              | 配置                   |
| --------------- | -------------------- |
| EP              | 8、64、128             |
| Tokens per Rank | 512、2048、4096        |
| Hidden Size     | 7168                 |
| FFN Size        | 1024、2048、3072       |
| Chunk 数         | 1、2、4、8              |
| 通信资源分配          | 支持配置的 AIV 核数量或通信引擎资源 |
| 执行方式            | 串行、多流、分块流水、单段融合、整层融合 |
| dtype           | BF16                 |
| 测试层次            | L0、L2、L3             |

重点分析计算通信比变化时，不同融合技术的适用范围。

通过改变 FFN Size、Token 数和 EP 规模，主动构造不同计算通信比例的场景，而非只测试单个固定模型 Shape。

## 六、不同 EP 算子的测试覆盖矩阵

并不是所有 EP 实现都应该参加全部测试。建议在 Backend 注册时声明其能力，根据能力自动生成测试用例。

| Backend                 | L1 纯通信 | L2 单段融合 | L3 完整前向 | L3 完整反向 | L4 负载均衡 |
| ----------------------- | ------ | ------- | ------- | ------- | ------- |
| HCCL AllGather          | ✓      | 基线      | ✓       | ✓       | —       |
| HCCL AllToAllV          | ✓      | 基线      | ✓       | ✓       | —       |
| DeepEP-Ascend           | ✓      | 基线      | ✓       | 适配      | —       |
| DeepEP CANN             | ✓      | 基线      | ✓       | 适配      | —       |
| DeepEP lueelu           | ✓      | 基线      | ✓       | 适配      | —       |
| MoonEP                  | ✓      | 基线      | ✓       | 适配      | ✓       |
| CANN MC2 Dispatch       | ✓      | 基线      | ✓       | 适配      | —       |
| CANN AllToAllV GMM      | —      | ✓       | ✓       | 适配      | —       |
| FSDPTurbo MC2           | —      | ✓       | ✓       | ✓       | —       |
| FSDPTurbo Domino        | 基线     | 流水      | ✓       | ✓       | —       |
| FSDPTurbo ChunkMoE      | 基线     | 流水      | ✓       | ✓       | —       |
| MindSpeed Overlap       | 基线     | 流水      | ✓       | ✓       | —       |
| EP_GMM_FUSED            | —      | ✓       | ✓       | 适配      | —       |
| MegaMoE lueelu          | —      | —       | ✓       | ✓       | —       |
| MegaMoE FSDPTurbo       | —      | —       | ✓       | ✓       | —       |
| MegaMoE TD              | —      | 视接口     | ✓       | ✓       | —       |
| MegaMoE CANN            | —      | 视接口     | 量化      | 待验证     | —       |
| DeepGEMM-Ascend         | —      | 视接口     | 量化      | 待验证     | —       |
| vLLM DispatchFFNCombine | —      | —       | ✓       | —       | —       |
| SGLang FuseEP           | —      | —       | ✓       | —       | —       |

说明：此表是根据之前 EP 算子调研形成的初始测试规划，不代表已经验证所有 Backend 均支持 EP64/128。正式注册时必须根据具体 Commit、硬件、接口、dtype 和拓扑验证能力。

## 七、Benchmark 执行规范

### 7.1 标准测试流程

1. 初始化运行环境

   初始化通信域、绑定设备、记录硬件和软件版本。
2. 生成统一 Workload

   构造 Hidden States、Routing Trace、专家权重和输出梯度。
3. 正确性验证

   验证专家归属、Dispatch/Combine、完整输出和训练梯度。
4. 初始化通信 Buffer

   分配 SHMEM、Workspace、Handle 等资源，单独记录初始化耗时。
5. Warmup

   排除首次 JIT 编译、通信域建立、缓存初始化等冷启动影响。
6. 正式性能采样

   关闭 Profiling，采集 Host 和 Device 耗时。
7. 独立 Profiling 采样

   使用相同配置采集 NPU、CPU、通信和算子时间线。
8. 多 Rank 结果聚合

   计算 Rank Max、P50/P95/P99、带宽、吞吐和资源消耗。
9. 自动生成性能报告

   记录性能排名、瓶颈分析、失败原因和原始数据索引。

### 7.2 采样次数

| 测试类型           | Warmup | 正式采样       | 独立重复运行 |
| -------------- | ------ | ---------- | ------ |
| L0 硬件基线        | 50     | 500        | 3      |
| L1 训练尾块延迟 | 100    | 2000       | 3      |
| L1 大 Batch 高吞吐 | 50     | 200        | 3      |
| L2 通算融合        | 20     | 100        | 3      |
| L3 完整 MoE      | 20     | 100        | 3      |
| L4 动态负载均衡      | 20     | 至少覆盖多个规划周期 | 3      |
| Profiling      | 1–5    | 3–10       | 独立执行   |

这些是建议的初始采样参数。大规模 EP 测试可以根据测试成本调整，但 P99 需要足够多的样本，不建议仅使用 50 次采样就得出稳定的 P99 结论。

### 7.3 计时逻辑

同步执行模式可以采用以下公共逻辑：

```python
for iteration in range(warmup + iterations):
    prepare_inputs()
    reset_gradients()
    torch.npu.synchronize()
    dist.barrier()
    torch.npu.synchronize()

    start = time.perf_counter_ns()
    output = backend.execute(workload)
    backend.wait_until_ready(output)
    torch.npu.synchronize()
    end = time.perf_counter_ns()

    if iteration >= warmup:
        record_latency(end - start)
```

需要注意：

| 操作                      | 是否计入正式耗时                   |
| ----------------------- | -------------------------- |
| 通信域初始化                  | 否，单独统计                     |
| JIT 编译                  | 否，单独统计                     |
| 随机输入生成                  | 否                          |
| 初始通信 Buffer 分配          | 否，单独统计                     |
| 每次调用必需的 Buffer 分配       | 是                          |
| 测试框架额外 Barrier          | 否                          |
| 算子内部必需的同步               | 是                          |
| 路由 Metadata 生成          | Full 模式包含，Cached 模式按接口语义执行 |
| Token Permute/Unpermute | 是                          |
| 异步通信等待                  | 是，完整 Ready Latency 必须包含    |
| Profiling 额外开销          | 不纳入正式性能排名                  |

此外，还应该区分三种执行模式：

- Single-call Latency：每次执行等待完成，测量独立调用延迟。
- Steady-state Throughput：连续执行多次，允许合法异步流水，测量持续吞吐。
- Async Issue + Wait：分别记录提交、计算重叠、等待和结果可用时间。

不能用每个算子后都强制 `synchronize()` 的方式评价通算融合，因为这可能破坏 Backend 原生的异步流水。

### 7.4 正确性验证

正确性必须作为性能排名的前置条件。

| 验证项                | 验证方式                             |
| ------------------ | -------------------------------- |
| Dispatch Token 数   | 与 Routing Trace 中专家计数一致          |
| Dispatch 数据        | 根据源 Token、Expert ID 验证           |
| Combine 输出         | 与 FP32 Reference 对比              |
| 完整 Forward         | 与非融合专家计算参考实现对比                   |
| Input Grad         | 验证输入梯度                           |
| Expert Wgrad       | 验证 Gate、Up、Down 权重梯度             |
| Router Weight Grad | 验证路由权重梯度                         |
| Dedup 正确性          | 去重开关前后的逻辑结果一致                    |
| 冗余专家正确性            | 验证专家映射、权重版本及梯度归并                 |
| 多次调用               | 验证 Buffer 复用、缓存 Handle 与动态路由     |
| 极端输入               | 零 Token Rank、空专家、容量边界、不等 Token 数 |

BF16、FP8 等不同 dtype 应定义不同的误差容忍范围，并基于参考实现和数值实验确定。

错误结果、丢 Token、计算语义不一致的 Backend 即使速度最快，也不得进入有效性能排名。

## 八、如何根据测试指标定位性能瓶颈

以下是整套 Benchmark 最终需要具备的分析能力。

| 现象                   | 可能瓶颈           | 进一步检查指标                      |
| -------------------- | -------------- | ---------------------------- |
| Dispatch 慢、带宽低       | 通信数据面效率低       | URMA/UDMA、MTE、实际流量           |
| Dispatch 慢、设备空闲高     | Host 调度或同步     | CPU Trace、D2H/H2D、Issue      |
| 通信快、完整 MoE 慢         | 专家计算效率低        | GMM TFLOPS、Padding、重排        |
| EP8 快、EP128 慢        | 扩展能力不足         | 跨 Server 流量、拓扑拥塞             |
| P50 正常、P99 很高        | 长尾、慢 Rank      | Rank Max/Avg、同步等待            |
| Forward 快、Backward 慢 | 反向 GMM 或梯度通信   | Dgrad、Wgrad、逆向通信             |
| 通算重叠率高、整体收益低         | 硬件资源竞争         | AIV/AIC/HBM 并发干扰             |
| 去重流量下降、性能未提升         | Metadata 或转发开销 | 去重准备、Epilogue                |
| Planner 后负载均衡但性能下降   | 规划和权重交换成本      | Planner、Prefetch、Reduce Grad |
| 性能快、显存占用高            | 大 Buffer 或冗余专家 | Workspace、SHMEM、权重副本         |

### 8.1 以现有 EP8 结果为例

此前的 Benchmark 有两个特别值得进一步分析的 Backend。

#### 现有 EP8 性能数据

| Backend | 完整专家子层前反向 P50（ms） | 设备空闲时间（ms） |
| --- | ---: | ---: |
| `megamoe_lueelu` | 25.97 | 0.72–0.91 |
| `deepep_ascend` | 32.74 | 0.67–0.77 |
| `alltoall_hccl` | 37.05 | 1.00–1.32 |
| `moonep` | 56.42 | 20.83–21.68 |
| `deepep_lueelu` | 71.67 | 28.58–37.81 |

来源：[ascendep EP8 Benchmark](https://github.com/penn513/ascendep/blob/main/reports/ep8-profile/benchmark.md?utm_source=chatgpt.com)。设备空闲来自独立 Profiling 采样，不应直接从前反向 P50 中相减。

通过当前数据只能发现 `moonep` 和 `deepep_lueelu` 存在明显的设备空闲现象，无法确定其通信数据面是否比其他通信实现更慢。

新 Benchmark 可按本节的瓶颈对照逐项分析。

这才是分层 Benchmark 相比现有完整 MoE 耗时测试最有价值的地方：不仅能够排名，还能够逐步排除不同的性能瓶颈。

## 九、测试脚本目录设计

结合前面讨论的 `infra/common/ep/` 目录，建议将详细测试设计放入 `03_benchmark`，具体实现放入 `04_scripts`。

```text
infra/common/ep/
│
├── 03_benchmark/
│   ├── ep_benchmark_design.md
│   ├── ep_benchmark_methodology.md
│   ├── ep_benchmark_metrics.md
│   └── ep_benchmark_testcases.md
│
├── 04_scripts/
│   ├── README.md
│   ├── run_benchmark.py
│   │
│   ├── configs/
│   │   ├── ep8_baseline.yaml
│   │   ├── large_token_training.yaml
│   │   ├── high_throughput.yaml
│   │   ├── weak_scaling.yaml
│   │   ├── strong_scaling.yaml
│   │   ├── topology.yaml
│   │   ├── load_balance.yaml
│   │   ├── dedup.yaml
│   │   └── overlap.yaml
│   │
│   ├── workloads/
│   │   ├── tensors.py
│   │   ├── routing_balanced.py
│   │   ├── routing_random.py
│   │   ├── routing_hotspot.py
│   │   ├── routing_locality.py
│   │   ├── routing_dedup.py
│   │   └── routing_trace.py
│   │
│   ├── backends/
│   │   ├── base.py
│   │   ├── registry.py
│   │   ├── hccl/
│   │   ├── deepep/
│   │   ├── moonep/
│   │   ├── cann_mc2/
│   │   ├── megamoe/
│   │   └── frameworks/
│   │
│   ├── benchmarks/
│   │   ├── l0_hardware.py
│   │   ├── l1_communication.py
│   │   ├── l2_fused.py
│   │   ├── l3_expert.py
│   │   └── l4_moe_layer.py
│   │
│   ├── metrics/
│   │   ├── latency.py
│   │   ├── bandwidth.py
│   │   ├── throughput.py
│   │   ├── flops.py
│   │   ├── overlap.py
│   │   ├── load_balance.py
│   │   ├── scalability.py
│   │   └── memory.py
│   │
│   ├── validation/
│   │   ├── reference.py
│   │   ├── forward.py
│   │   ├── backward.py
│   │   └── routing.py
│   │
│   ├── profiling/
│   │   ├── collector.py
│   │   ├── parser.py
│   │   ├── communication.py
│   │   └── overlap.py
│   │
│   ├── reports/
│   │   ├── schema.py
│   │   ├── aggregate.py
│   │   └── generate_report.py
│   │
│   ├── launchers/
│   │   ├── run_ep8.sh
│   │   ├── run_ep64.sh
│   │   └── run_ep128.sh
│   │
│   └── tests/
│
└── 05_results/
    ├── ep_benchmark_ep8.md
    ├── ep_benchmark_ep64_ep128.md
    └── raw/
```

测试脚本通过 Backend Registry 管理能力，通过 Workload Generator 统一构造输入，通过 Benchmark Runner 控制测试范围，Metrics 模块负责公共指标计算。

例如最终期望支持如下命令：

```bash
# EP8 完整基线
python run_benchmark.py \
    --config configs/ep8_baseline.yaml

# EP128 纯通信对比
python run_benchmark.py \
    --config configs/high_throughput.yaml \
    --ep-size 128 \
    --level L1 \
    --backends deepep_ascend,moonep,mc2_dispatch_cann

# 通算融合对比
python run_benchmark.py \
    --config configs/overlap.yaml \
    --level L2,L3

# 动态专家负载均衡专项测试
python run_benchmark.py \
    --config configs/load_balance.yaml \
    --ep-size 128
```

这些是建议的 CLI 接口设计，不是当前仓库已经具备的可执行命令。

## 十、最终报告及算子评价体系

### 10.1 核心性能结果表

最终每个 Backend 不仅有一个 tokens/s 排名，而是有多个分层性能表。

通信性能表：

| Backend  | Dispatch P50 | Combine P50 | Payload GB/s | 基线带宽效率 | 通信量 |
| -------- | ------------ | ----------- | ------------ | ------ | --- |
| HCCL     | 待测           | 待测          | 待测           | 待测     | 待测  |
| DeepEP   | 待测           | 待测          | 待测           | 待测     | 待测  |
| MoonEP   | 待测           | 待测          | 待测           | 待测     | 待测  |
| CANN MC2 | 待测           | 待测          | 待测           | 待测     | 待测  |

通算融合性能表：

| Backend  | 串行基线 | 融合耗时 | 加速比 | 可见重叠率 | GMM 效率 |
| -------- | ---- | ---- | --- | ----- | ------ |
| Domino   | 待测   | 待测   | 待测  | 待测    | 待测     |
| ChunkMoE | 待测   | 待测   | 待测  | 待测    | 待测     |
| MC2      | 待测   | 待测   | 待测  | 待测    | 待测     |
| MegaMoE  | 待测   | 待测   | 待测  | 需内部分析 | 待测     |

大规模扩展性能表：

| Backend       | EP8 TPS | EP64 TPS | EP128 TPS | EP128 扩展效率 |
| ------------- | ------- | -------- | --------- | ---------- |
| DeepEP-Ascend | 待测      | 待测       | 待测        | 待测         |
| MoonEP        | 待测      | 待测       | 待测        | 待测         |
| MC2           | 待测      | 待测       | 待测        | 待测         |
| MegaMoE       | 待测      | 待测       | 待测        | 待测         |

### 10.2 综合评价权重

不同训练场景可按实际瓶颈调整综合评价权重。

下面可以作为初始评价体系。

#### 不同业务场景的评价指标权重

下表给出训练场景的初始权重示例。

| 指标 | 初始权重 |
| --- | ---: |
| 通信延迟 | 10% |
| 业务吞吐 | 30% |
| 通算融合收益 | 15% |
| EP 扩展效率 | 20% |
| 显存与资源 | 10% |
| 负载均衡收益 | 10% |
| 长尾稳定性 | 5% |

建议的初始权重，后续可根据业务实际瓶颈调整。不能将不支持的必需能力简单记为零分后继续参与综合排名。

正确性、硬件支持、显存容量和必需功能是准入条件，而不是可以通过其他性能得分弥补的指标。

对于满足准入条件的 Backend，可以在同一场景中相对统一基线计算归一化收益，再结合权重或 Pareto 前沿进行选型。

### 10.3 建议的性能验收标准

以下阈值可以作为第一阶段的内部目标，并不是行业标准或已经验证的硬件性能要求。

| 指标                 | 建议初始目标                    |
| ------------------ | ------------------------- |
| 正确性                | 全部必需测试通过                  |
| 大消息通信带宽            | 达到同口径 L0 实测基线的 80%        |
| 独立 GMM 效率          | 达到对应 Shape 实测 GMM 基线的 80% |
| 通算融合收益             | 相同语义下快于非融合串行基线            |
| EP64 Weak Scaling  | ≥ 80%                     |
| EP128 Weak Scaling | ≥ 70%                     |
| 高吞吐 P95/P50        | ≤ 1.10                    |
| 去重方案               | 流量下降且端到端性能提升              |
| 动态负载均衡             | 计入全部额外开销后仍有正收益            |
| 显存消耗               | 满足实际训练资源预算             |

这些阈值需要在第一轮 L0 和 EP8 测试后修订，尤其是跨 Server 拓扑条件下的 EP64/128 扩展效率。

## 十一、实施优先级

| 阶段       | 工作内容                                 | 优先级 |
| -------- | ------------------------------------ | --- |
| Phase 1  | 统一 Workload、Routing Trace、Backend 接口 | P0  |
| Phase 2  | L0 硬件通信和 GMM 基线                      | P0  |
| Phase 3  | L1 Dispatch/Combine 纯通信测试            | P0  |
| Phase 4  | 复用 ascendep 的 L3 前反向测试               | P0  |
| Phase 5  | L2 通算融合及非融合对照测试                      | P0  |
| Phase 6  | EP64/128 Weak/Strong Scaling         | P0  |
| Phase 7  | Profiling、带宽、设备空闲归因                  | P0  |
| Phase 8  | MoonEP 动态负载均衡及去重专项                   | P1  |
| Phase 9  | 量化训练、大 Token 数及真实训练路由 | P1  |
| Phase 10 | L4 完整业务评估与自动化选型报告                    | P1  |

## 最终总结

这套 Benchmark 最核心的设计是将 EP 算子的评价拆成五个问题：

| 问题                 | 对应测试 |
| ------------------ | ---- |
| 硬件本身能达到什么性能？       | L0   |
| 通信算子本身有多快？         | L1   |
| 通信与计算融合能节省多少时间？    | L2   |
| 完整专家计算的业务性能如何？     | L3   |
| 在真实路由、动态负载下是否仍然有效？ | L4   |

对于当前的昇腾 EP64/128 算子调研，我建议第一阶段重点建设 L0 + L1 + L2 + L3，优先解决纯通信性能、通算融合收益以及大规模扩展性三个问题。

现有 ascendep 的 Benchmark 及 Backend Adapter 可以作为 L3 的起点，而没有必要重新实现全部 24 个后端。

以上方案属于详细设计，尚未实际修改 `infra` 仓库，也未执行新的 NPU 性能测试。文中的测试矩阵、验收阈值和评价权重均为建议配置，而不是已取得的实测结论。
