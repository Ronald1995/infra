# AIV 直搬路径：MTE 驱动的 Dispatch / Combine

> **定位**：本文只讨论「Payload 由 AIV 的 MTE 搬运」这一条路径，典型平台是 A3 超节点（CloudMatrix384 一类支持远端内存映射的系统）。如果实现是「AIV 提交任务、UDMA 搬数据」，看 [jetty_udma.md](jetty_udma.md)。
>
> **一句话结论**：AIV 不只是向量计算单元，它同时是一套可编程的细粒度搬运资源。理解 Dispatch / Combine 的性能，必须同时看 MTE2、MTE3、UB、Vector、Scalar 五条资源，而不是只看跨卡带宽。

## 目录

- [一、AIV 不是向量计算单元](#一aiv-不是向量计算单元)
- [二、AIV 如何访问远端内存](#二aiv-如何访问远端内存)
- [三、Dispatch 的完整数据流](#三dispatch-的完整数据流)
- [四、Combine 的完整数据流](#四combine-的完整数据流)
- [五、瓶颈在哪里](#五瓶颈在哪里)
- [六、流水重叠如何实现](#六流水重叠如何实现)
- [七、与引擎搬运路径的分界](#七与引擎搬运路径的分界)
- [八、Profiling 判据](#八profiling-判据)
- [九、小结](#九小结)
- [参考资料](#参考资料)
- [证据等级](#证据等级)

## 一、AIV 不是向量计算单元

在昇腾上，AIV（AI Vector Core）是一类可编程计算核，内部由多个硬件模块组成。做通信时，这些模块各有分工：

| 硬件模块 | 含义 | 在 Dispatch / Combine 中的作用 |
|---|---|---|
| Scalar | 标量控制单元 | 计算地址、Token 路由、控制循环、下发指令 |
| Vector | 向量计算单元 | 量化、反量化、乘法、加权求和 |
| UB | Unified Buffer，核内 SRAM | 暂存 Token、路由信息、量化结果、聚合中间结果 |
| MTE2 | Memory Transfer Engine 2 | 把 GM 数据搬入 UB |
| MTE3 | Memory Transfer Engine 3 | 把 UB 数据搬出到 GM |
| GM | Global Memory | 全局内存地址空间，通常对应 HBM 中的张量与通信窗口 |
| AIC | AI Cube Core | 执行专家 FFN 的矩阵乘法 |
| HCCS / UB Bus | 跨芯片互联 | 承载跨 NPU 的数据传输 |

关键事实：**MTE2、MTE3 和 Vector 是相对独立的硬件流水线，可以异步并行执行**。这是昇腾实现「搬运与计算重叠」的硬件基础，也是后面流水化优化的前提。

### 1.1 为什么通信也要过 UB

因为 AIV 的普通 Vector 计算主要操作核内数据，而不是直接对 HBM 中的 Tensor 做逐元素计算。对一个 Token 做量化，典型路径是：

```text
GM 中的 BF16 Token
  → MTE2 搬入 UB
  → Vector 量化
  → UB 中的 INT8 Token
  → MTE3 搬回 GM（或直接写远端）
```

即使完全不做计算，只想把「内存中的一个张量搬到另一个地方」，走 AIV MTE 路径时通常也要经过 UB 中转：MTE2 读入、MTE3 写出。原因不是物理上做不到 GM 到 GM，而是**这条编程模型里的搬运动词就是 GM 与 UB 之间**。真正能绕开 UB 的是另一类引擎（UDMA、SDMA 等），它们可以直接在 GM 之间搬运，这正好是两条路径的分野。

所以 UB 不只是计算缓存，它同时是 AIV 通信路径上的中转缓冲区。这一条决定了后面很多瓶颈的来源。

## 二、AIV 如何访问远端内存

对于 A3 CloudMatrix384 等支持全局共享内存、内存语义访问的系统，AIV 的 MTE 不仅能访问本地 GM，也能通过映射后的远端地址访问其他 NPU 的 GM。

于是 MTE2 和 MTE3 组合出四种基本搬运路径：

| 搬运指令 | 源地址 | 目的地址 |
|---|---|---|
| MTE2 本地读 | 本地 GM | 本地 AIV UB |
| MTE2 远端读 | 远端 GM | 本地 AIV UB |
| MTE3 本地写 | 本地 AIV UB | 本地 GM |
| MTE3 远端写 | 本地 AIV UB | 远端 GM |

远端访问依赖硬件互联能力、地址映射和访问权限，**并不是任意两张 NPU 卡都天然支持**。CloudMatrix384 的 AIV-Direct 就是利用这种能力降低通信启动时延。

### 2.1 Push：主动向远端写入

假设 Rank 0 需要把一个 Token 发送到 Rank 1。Push 模式下，发送端的 AIV 主动用 MTE3 写入对端 GM：

```text
Rank 0 HBM
  → Rank 0 AIV MTE2
  → Rank 0 AIV UB
  → Rank 0 AIV MTE3
  → Rank 0 片上互联与 I/O
  → 跨 NPU 互联网络
  → Rank 1 片上互联与 I/O
  → Rank 1 HBM
```

几个必须记住的结论：

- Token 经过发送端 AIV 的 UB。
- Token 经过发送端 AIV 的 MTE2 和 MTE3。
- 远端写入本身**不需要**经过接收端 AIV 的 UB。
- 远端写入本身也**不需要**接收端 AIV 的 MTE2 / MTE3 参与。
- 接收端后续做重排或计算时，才需要自己的 AIV 去读接收缓冲区。

这就是单边通信（One-sided Communication）的典型场景：数据面只有一边在动，但建链阶段仍然是双方握手。

### 2.2 Pull：主动从远端读取

另一种方式是接收端主动拉取：Rank 1 的 AIV 用 MTE2 读取 Rank 0 的 GM。

| 对比项 | Push | Pull |
|---|---|---|
| 主动发起方 | 发送端 | 接收端 |
| 跨卡搬运单元 | 发送端 MTE3 | 接收端 MTE2 |
| 使用谁的 UB | 发送端 | 接收端 |
| 谁承担主要搬运压力 | 发送端 AIV | 接收端 AIV |
| 典型通信语义 | Put / Write | Get / Read |

所以分析 EP 通信性能时，第一步要看实现采用 Push、Pull 还是两者混合。不同选择会让不同 Rank 的 MTE2 / MTE3 负载严重不均衡，而 EP 恰恰是「有的 rank 收得多、有的 rank 发得多」的典型场景。

## 三、Dispatch 的完整数据流

假设三个 NPU，每个部署不同的专家；Rank 0 的 Token X 被路由到 Rank 1 和 Rank 2。为便于说明，设 hidden size = 7168、输入 BF16、Dispatch 时做 INT8 量化。

```text
Rank 0 HBM（Token，BF16）
  → MTE2 读入 UB
  → Vector 量化（INT8），Scalar 同时算目标 Rank 与偏移
  → MTE3 写远端通信窗口
  → 片内互联 → UB Bus / HCCS → Rank 1 / Rank 2 HBM
  → 接收端 AIV 轮询状态标记，确认数据就绪
  → 接收端 MTE2 读入 → UB → MTE3 写出到专家输入缓冲
  → AIC 执行专家矩阵乘法
```

### 3.1 每一步用到的硬件

| 阶段 | 操作 | 主要硬件 |
|---|---|---|
| 1 | 读取 Token、TopK IDs | MTE2、Scalar |
| 2 | 计算目标 Rank 和地址偏移 | Scalar / Vector |
| 3 | 将 Token 搬入 UB | MTE2 |
| 4 | Token 动态量化 | Vector |
| 5 | 写入远端通信窗口 | MTE3、跨卡互联 |
| 6 | 发布通信完成标记 | AIV、同步机制 |
| 7 | 等待接收数据就绪 | 接收端 AIV |
| 8 | 按 Expert 重排接收数据 | MTE2、UB、MTE3 |
| 9 | 执行专家矩阵乘法 | AIC，部分环节使用 AIV |

CloudMatrix-Infer 公开论文中的 FusedDispatch 采用了类似的数据发送、完成标记同步和接收数据组装流程，只是把三段做成了流水。

### 3.2 发送端的伪代码

下面这段只用于说明硬件流程，同步函数和量化函数都是伪代码，不是某个仓库的真实 API：

```c
// 1. 读取本地 Token（GM -> UB）
DataCopy(tokenUB, localTokenGM, count);

// 2. 等待 MTE2 完成
SyncMTE2ToVector();

// 3. 可选：动态量化（BF16 -> INT8）
Quantize(quantUB, tokenUB);

// 4. Scalar 计算目标 Rank 的 GM 地址
remoteAddr = peerWindowBase[dstRank] + offset;

// 5. 等待 Vector 完成
SyncVectorToMTE3();

// 6. MTE3 直接写入远端 GM
DataCopy(remoteAddr, quantUB, quantCount);

// 7. 数据可见后，发布完成标记
PublishFlagAfterRemoteWrite();
```

注意第 4 步：远端地址来自初始化阶段交换的窗口信息，不是在每次通信里临时协商。

### 3.3 SHMEM 通信为什么也涉及 UB

在 MoonEP 等 EP 实现里经常看到 SHMEM。要分清它解决什么问题：

- 如何分配可被其他 Rank 访问的通信窗口。
- 如何获得远端 Rank 的内存地址。
- 如何提供 Put、Get、同步等通信接口。

**SHMEM 本身不是一种独立的物理搬运引擎。** 它是一层抽象的对称内存与 RMA 接口，底下可以挂不同的数据通路：

```text
上层 Dispatch / Combine 算子
  → SHMEM API（分配窗口、取远端地址、Put / Get）
  → 具体搬运引擎（MTE、UDMA、RoCE、SDMA 之一）
  → 跨 NPU 互联
  → 远端 HBM
```

当 SHMEM 底层选择 MTE 路径时，实际搬运仍然经过 AIV 的 MTE 和 UB。当底层选择 UDMA 时，源码里会看到 `aclshmemx_udma_*` 一族的调用，Payload 就不必经过 UB。

这一区分很重要：**「用了 SHMEM」不能推断出「用了哪种搬运引擎」**，必须看它实际调用的接口和配置。

## 四、Combine 的完整数据流

Combine 与 Dispatch 最大的区别：

- Dispatch 是一个 Token 分发到多个专家。
- Combine 是多个专家的计算结果返回原始 Rank，并做加权求和。

假设 Rank 1 和 Rank 2 分别算出专家结果 Y1 和 Y2，路由权重为 0.3 和 0.7，最终结果为：

```text
Y = 0.3 * Y1 + 0.7 * Y2
```

接收端的典型执行过程：

```c
// 等待当前 Token 的所有专家结果就绪
WaitAllExpertResults();

// MTE2：读取专家结果（GM -> UB）
DataCopy(y1UB, recvExpert1GM, hiddenSize);
DataCopy(y2UB, recvExpert2GM, hiddenSize);

// Vector：加权求和
Muls(y1UB, y1UB, 0.3, hiddenSize);
Muls(y2UB, y2UB, 0.7, hiddenSize);
Add(resultUB, y1UB, y2UB, hiddenSize);

// MTE3：写回最终结果
DataCopy(outputGM, resultUB, hiddenSize);
```

这是简化示例，省略了流水同步、数据类型转换和跨专家分块。

因此 Combine 相比纯通信算子，往往额外背了两份压力：**Vector 的加权计算**和**UB 的访存占用**。当然，也有实现选择在发送端提前加权，或者做分层聚合，把压力前移或摊薄。

## 五、瓶颈在哪里

### 5.1 跨卡带宽高，不代表实际 Dispatch 带宽就高

假设所有 Token 都通过 AIV MTE 发送。对于不改变数据大小的单向搬运，忽略其他竞争时，有效带宽满足：

```text
B_effective <= min(B_HBM_read, B_MTE2, B_UB, B_MTE3, B_link, B_remote_write)
```

这里的 MTE 带宽指参与通信的多个 AIV 的**聚合有效带宽**。如果 MTE3 聚合后只能提供较低的发送带宽，即使超节点互联还有大量剩余带宽，通信也无法提速。

### 5.2 一个 Token 实际产生了多少访存

以 hidden = 7168、TopK = 8 为例，假设八个专家都在远端、每个专家一份直接 Push、忽略去重与元数据：

| 观测点 | 数据量 | 说明 |
|---|---|---|
| 本地 HBM 读（MTE2） | 14 KiB | 7168 × 2 B，BF16，单次读入 |
| UB 中单份待发数据 | 7 KiB | INT8 量化后 |
| 远端写（MTE3，累计） | 56 KiB | TopK = 8，每个专家一份 |
| 网络有效载荷（Dispatch） | 56 KiB | 不含协议头、元数据 |
| Combine 返回载荷 | 112 KiB | 八个专家 × 14 KiB，BF16 |

一个 Token 在本地只读一次，却可能要写八次。所以这种实现下，发送侧是「读一次、写八次」，接收侧还要把 56 KiB 再读入并重排一遍，双向的 MTE 负载都不轻，且读写比例严重不对称。

这也是 Token 去重、按 Rank 聚合、UB 数据复用能优化通信性能的根本原因：它们减少的是**写放大**，不是网络带宽。

### 5.3 UB 容量会限制通信流水

UB 是每个 AIV 核内有限的 SRAM，需要同时容纳：

- 原始 Token。
- 量化后的 Token。
- Scale 与路由信息。
- 临时计算数据。
- Ping-Pong 双缓冲。
- 同步标记及其他临时数据（视实现而定）。

UB 不够时只有两条路：切得更细（更多搬运指令、更多同步），或者降低并发度（流水变浅）。两者都会压低有效带宽。所以「UB 容量」是 AIV 直搬路径上一个容易被忽略的硬约束，调 kernel 时值得单独核对占用。

## 六、流水重叠如何实现

如果完全串行执行，一个 AIV 必须等当前 Token 发送完成，才能处理下一个 Token。用 UB 双缓冲后，MTE2、Vector、MTE3 可以同时处理不同的 Token：

| 时间片 | 串行执行 | 双缓冲流水 |
|---|---|---|
| 1 | MTE2 搬 T0 | MTE2 搬 T0 |
| 2 | Vector 处理 T0 | MTE2 搬 T1 |
| 3 | MTE3 发 T0 | MTE2 搬 T2、Vector 处理 T0 |
| 4 | MTE2 搬 T1 | MTE2 搬 T3、Vector 处理 T1、MTE3 发 T0 |

假设三个阶段耗时相同，且缓冲区可以安全复用。理想稳定流水中，每个 Token 的处理间隔由最慢的阶段决定：

```text
T_interval ≈ max(T_MTE2, T_Vector, T_MTE3)
```

而不是三者之和。总时间近似为「流水填充时间 + Token 数 × T_interval」。

实际流水效率取决于数据依赖、UB 分配和硬件竞争。CloudMatrix-Infer 的 FusedDispatch 也采用了「Token 搬入、地址计算与量化、远端写入」的三级流水设计，思路与此一致。

## 七、与引擎搬运路径的分界

AIV 发起通信，不代表 Payload 一定由 AIV 的 MTE 搬运。这两件事必须分开看：

```text
方式一：AIV 既控制又搬运
  AIV Scalar 算地址 → MTE2 读 → UB → MTE3 写远端

方式二：AIV 只控制，引擎搬运
  AIV Scalar 构造 WQE → 提交 Jetty → UDMA 从 GM 直接写远端
```

在方式二里，AIV 主要负责准备通信描述符（WQE）、提交请求、处理完成状态；Payload 由独立通信硬件从 GM 直接搬到远端 GM，不经过 UB，也不占用 MTE。

| 对比项 | AIV MTE 直搬 | AIV 直驱 RDMA / UDMA | SDMA |
|---|---|---|---|
| 实际搬运硬件 | MTE2 / MTE3 | 通信引擎 | SDMA 引擎 |
| Payload 必经 AIV UB | 是，典型 MTE 路径 | 否 | 否 |
| AIV 负载 | 数据搬运及计算 | 提交、同步及可选的前后处理 | 视控制方式而定 |
| 小包通信 | 适合细粒度低时延场景 | 取决于请求提交效率 | 启动开销可能较明显 |
| UB 容量限制 | 明显 | 主要影响前后处理 | 不直接限制纯搬运 |
| 主要瓶颈 | MTE、UB、互联 | WQE 下发、SQ 深度、链路 | DMA、内存及链路 |

方式二的下发环节本身也可能成为瓶颈，详见 [jetty_udma.md](jetty_udma.md) 第六节。

## 八、Profiling 判据

结合 EP 算子在 64 / 128 Rank 下的常见测试，建议按硬件维度拆开看：

| 硬件维度 | 关注指标 | 判断目的 |
|---|---|---|
| AIV | 核数、执行周期、核间负载差异 | 通信核分配是否合理 |
| MTE2 | Active Cycles、读带宽 | Token 搬入是否受限 |
| MTE3 | Active Cycles、写带宽 | 远端写入是否受限 |
| Vector | Active Cycles、计算流水占比 | 量化或 Combine 聚合是否瓶颈 |
| UB | 空间占用、分块大小、双缓冲配置 | 流水能力是否受限 |
| Scalar | 执行周期、地址计算与提交开销 | 小包控制开销是否过大 |
| HBM | 读写带宽 | 通信与 GEMM 是否争抢内存 |
| HCCS / UB Bus | 实际吞吐、链路利用率 | 物理互联是否瓶颈 |
| RDMA / UDMA | Issue、Completion、Drain 延迟 | 独立通信引擎是否瓶颈 |
| 同步 | Flag polling、Barrier 等待 | 慢 Rank 与负载不均衡 |

具体计数器名称与可采集性以对应芯片和 msprof 版本为准。

**一个简单有效的实验：改 AIV 数量，看带宽是否继续增长。**

- 如果增加 AIV 明显提升带宽：当前受 AIV、MTE 或请求下发能力限制。
- 如果增加 AIV 后带宽不再增长：已经撞到互联、HBM、共享访存通路或通信引擎的上限。

但要同时看副作用：增加通信 AIV 数量可能提升独立 Dispatch 性能，却挤占了专家计算要用的 Vector 核，整体 MoE 通算重叠效率反而下降。这正是官方文档对 AIV 引擎的描述——「通信占用 Vector 计算核，需要多个 Vector 计算核才能打满通信带宽；通信算子与计算算子竞争计算核资源，可能互相影响」。

## 九、小结

| 操作 | 典型硬件路径（AIV MTE Push 实现） |
|---|---|
| Dispatch 发送 | HBM → MTE2 → UB → 可选量化 → MTE3 → 远端 HBM |
| Dispatch 接收重排 | HBM → MTE2 → UB → MTE3 → 专家输入 HBM |
| 专家计算 | HBM → AIC 内部 Buffer → Cube → 输出 HBM |
| Combine 发送 | HBM → MTE2 → UB → MTE3 → 源 Rank HBM |
| Combine 聚合 | HBM → MTE2 → UB → Vector 加权求和 → MTE3 → HBM |

这条路径的核心理解是：

- UB 是数据搬运和计算的中转站。
- MTE2 是读入通路，MTE3 是写出通路。
- Vector 是数据处理单元，Scalar 是控制单元。
- HCCS / UB Bus 是跨芯片传输通路。

因此优化昇腾上的 Dispatch 和 Combine，不能只盯跨卡网络带宽，还要同时优化：AIV 核间负载均衡、UB 数据复用、MTE 搬运效率、同步开销，以及通算流水重叠。

## 参考资料

- [Ascend C：AI Core 硬件架构及数据搬运路径](https://asc.gitcode.com/api/SIMD-API/c_api/data_move_guide/overview/data_path.html)
- [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/html/2506.12708v3)，§4.2.1，重点看 Figure 11、12
- [Huawei Cloud Model-as-a-Service on the CloudMatrix384 SuperPod](https://arxiv.org/html/2508.02520v6)，§3 通信库设计
- [CANN：MoE Dispatch & Combine 通算融合算子](https://www.hiascend.com/developer/techArticles/20250726-1)
- [昇腾开源 SHMEM 通信库](https://github.com/hicann/shmem)
- [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)

本机 CANN 知识库中的对应材料：

- 通信引擎对比（AIV 低时延、占 Vector 核、需多核打满带宽）—— `cann-docs-raw/hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_engine.md`
- AIV 作为发起引擎、基于 hcomm + UDMA 的通信接口 —— `cann-docs-raw/ops-transformer/mc2/common/op_kernel/apace/README.md`
- MegaMoE 的 `topoType`（0 为 MTE 拓扑、1 为 URMA 跨超拓扑）—— `cann-docs-raw/ops-transformer/mc2/mega_moe/README.md`
- SHMEM 的地址与引擎边界（MTE / SDMA / UDMA / RDMA 不可互换）—— `knowledge/ops/ascendc/guides/shmem/symmetric_memory_and_completion.md`

## 证据等级

- **官方文档明确记载**：AIV 作为通信引擎的定位与约束、AIV 直驱接口的存在、`topoType` 的两种通信拓扑、Dispatch / Combine 算子的 `commType` 取值。这些是本机 CANN 知识库归档文档中的原文结论。
- **论文与源码观察**：CloudMatrix384 的 AIV-Direct 与 FusedDispatch 流程、远端写入与标志位同步的具体实现细节。属于公开论文与开源实现的描述，版本间可能有差异。
- **工程推断**：14 KiB / 7 KiB / 56 KiB / 112 KiB 的访存量估算、`T_interval` 模型、AIV 扩容实验的判读方式。这些是便于分析的量级估算，未做硬件实测，不能当性能承诺。

---

*路径 B（AIV 提交、UDMA 搬运）见 [jetty_udma.md](jetty_udma.md)；两条路径的统一对照见 [README.md](README.md)。*
