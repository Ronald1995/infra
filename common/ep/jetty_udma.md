# Jetty + UDMA：URMA 通信栈与 RDMA QP 的对照

> **定位**：本文讨论「AIV 只提交任务、由通信引擎搬 Payload」这条路径，典型平台是 Ascend 950PR / 950DT（A5）。如果实现是 AIV 自己用 MTE 搬，看 [aiv_mte.md](aiv_mte.md)。
>
> **一句话结论**：Jetty 是通信任务的提交端点，UDMA 是实际搬数据的硬件引擎，URMA 是统一的远端内存访问机制与接口，UB 是承载跨芯片传输的互联体系，AIV 是组织任务、提交请求、处理同步的可编程计算核。它们属于同一套通信体系的不同层次，不是互相并列的几种硬件。

## 目录

- [一、几个通信概念的关系](#一几个通信概念的关系)
- [二、Jetty 与 RDMA QP](#二jetty-与-rdma-qp)
- [三、一次 Jetty 通信的硬件流程](#三一次-jetty-通信的硬件流程)
- [四、为什么代码里还有 UB 和 MTE3](#四为什么代码里还有-ub-和-mte3)
- [五、为什么要多个 Jetty](#五为什么要多个-jetty)
- [六、Dispatch 与 Combine 的阶段化性能模型](#六dispatch-与-combine-的阶段化性能模型)
- [七、三种通信方式对比](#七三种通信方式对比)
- [八、平台边界与常见误解](#八平台边界与常见误解)
- [参考资料](#参考资料)
- [证据等级](#证据等级)

## 一、几个通信概念的关系

先划清层次，这是理解后面所有内容的前提：

| 概念 | 物流类比 | 实际作用 |
|---|---|---|
| AIV | 发货人员 | 组织数据、填写通信任务 |
| WQE | 运输订单 | 描述源地址、目的地址、数据长度等 |
| Jetty | 物流业务端点 | 管理通信任务及其队列 |
| SQ | 待发货订单队列 | 保存待执行的 WQE |
| Doorbell | 通知发货 | 通知硬件有新任务 |
| UDMA | 运输车辆 | 实际搬运数据 |
| UB Bus | 高速公路 | 承载跨芯片传输 |
| CQE | 运输完成回执 | 表示对应任务已完成 |

所以「Jetty 通信」本质上就是：**AIV 通过 Doorbell 向 URMA 的 Jetty 队列提交任务，由 UDMA 硬件执行跨 NPU 的内存搬运**。

再加两个容易漏掉的：

- **URMA（Unified Remote Memory Access）** 是这套远端内存访问的机制与编程接口，Jetty、WQE、CQE 都是它的概念。
- **UB（Unified Bus）** 是 950 的互连底座。注意它和 AIV 核内的 **UB（Unified Buffer）** 没有关系，只是缩写撞车。

950 白皮书对 URMA 的描述是：异步模式使用 Jetty 的 SQE / CQE，支持 Write / Read / Send / Atomic；RTP 模式 4 个端口可靠传输，CTP 模式 9 个端口。这也解释了为什么 Jetty 会成为 950 通信栈的核心对象。

## 二、Jetty 与 RDMA QP

Jetty 可以类比传统 RDMA 的 QP（Queue Pair），但两者并不完全等价。把两套栈的对应层并排放：

```text
RDMA / RoCE           URMA / UB
RDMA Verbs            URMA API
QP                    Jetty
RDMA Engine           UDMA Engine
RoCE / InfiniBand     Unified Bus
```

逐项对照：

| 对比项 | RDMA / RoCE | URMA / UB |
|---|---|---|
| 通信引擎 | RDMA Engine | UDMA Engine |
| 通信端点 | QP | Jetty |
| 发送队列 | SQ | SQ / JFS |
| 接收队列 | RQ | RQ / JFR |
| 完成队列 | CQ | CQ / JFC |
| 通信任务 | WQE | WQE |
| 硬件通知 | Doorbell | Doorbell |
| 完成事件 | CQE | CQE |
| 内存管理 | MR | Segment |
| 内存访问凭证 | lkey / rkey | Token 等 |
| 网络端点标识 | GID 等 | EID |
| 通信网络 | RoCE / IB | Unified Bus |

角色是一一对应的，这也是「Jetty 像 QP」这个说法成立的原因。但有三点不能直接套用：

1. **抽象层次不同。** URMA 把通信端点、内存访问对象（Segment）和底层传输通道拆成三套独立概念，Jetty 不等于「连接」。而 QP 在 RDMA 里同时绑定了连接与队列语义。
2. **传输模式不止一种。** 公开材料提到 URMA 支持多种传输模式（RC、RM、UM 一类），具体定义与可用组合以 UB / URMA 规范为准，本文未逐一核对。因此不要照搬「QP 一定是 RC」这类 RDMA 经验。
3. **上层对象不同。** HCCL 侧看到的是 Channel（= 两端 Endpoint + 协议 + N 个 Notify），Jetty 是 Channel 在 UB 协议下的底层载体，默认 1:1，但多通道、共享 Jetty 等模式会打破这个比例。详见 [../../UB/Jetty.md](../../UB/Jetty.md)。

## 三、一次 Jetty 通信的硬件流程

假设 Rank 0 的 AIV 要通过 Jetty 把一个 Token 写入 Rank 1 的 HBM。

### 3.1 初始化：建立通信资源

这部分通常在 Host 侧完成，且只做一次：

```text
创建通信上下文
  → 注册本地通信内存 Segment
  → 创建 Jetty / SQ / CQ
  → 交换远端 EID、Jetty、Segment 信息
  → 建立或配置传输通道
  → 将通信上下文提供给 AIV
```

这些操作不会在每次 Dispatch 时重做，而是初始化阶段建好、后续复用。这也是为什么通信启动开销在 warmup 之后通常可以忽略，而「每轮重新建链」的实现性能会很差。

Jetty 可以按实现分配给不同 AIV 使用，也可以在满足并发访问约束的条件下共享。

### 3.2 运行期：AIV 提交 WQE

WQE 是通信任务描述符。下面是原理示意，不是真实的硬件结构体：

```c
// 原理示意，并非实际硬件结构体
struct WQE {
    uint64_t src_addr;
    uint64_t dst_addr;
    uint32_t length;
    uint32_t opcode;       // WRITE / READ / AMO
    uint32_t dst_jetty;
    uint32_t dst_eid;
    uint32_t token;
};
```

要点：**WQE 里放的是地址、长度、操作类型这类描述信息，不是 Token 数据本身**。Token 还在本地 HBM 里，等着 UDMA 去取。

### 3.3 控制流与数据流

这是全文最值得记住的一张图：一次 Jetty 通信有两条完全不同的数据流。

控制流（走 AIV 的 Scalar 和片上总线）：

```text
AIV Scalar
  → 构造 WQE
  → 写入 Jetty 的 SQ（可能在 GM，也可能先经 UB staging）
  → 推进 PI
  → 敲 JFS Doorbell
  → UDMA Engine 取走并解析任务
```

实际 Token 数据流：

```text
本地 HBM
  → 片上访存互联
  → UDMA Engine
  → UB Port
  → 跨芯片互联（Unified Bus）
  → 远端访存系统
  → 远端 HBM
```

关键结论：**实际 Token 数据不经过 AIV 的 UB、MTE2 和 MTE3**。AIV 只碰控制信息，Payload 由 UDMA 从 GM 直接搬到远端 GM。

这正是 Jetty + UDMA 相对于 AIV MTE 直接搬运的核心区别：控制面与数据面解耦。

## 四、为什么代码里还有 UB 和 MTE3

这是最容易产生「既然 UDMA 搬数据，为什么 kernel 里还有 MTE3」疑问的地方。

因为 UDMA 只负责搬 Payload，**准备通信任务本身也需要内存搬运**。CANN SHMEM 的 UDMA 实现里就有这样的做法：用 UB 拼装 WQE，再通过 MTE3 把 WQE 搬到 GM 中的发送队列，并为此提供 UB staging 配置。

于是 MTE 在两种通信方式里的职责完全不同：

| 对比项 | 方式 A：AIV MTE 通信 | 方式 B：Jetty + UDMA 通信 |
|---|---|---|
| MTE3 搬什么 | 完整 Token Payload | WQE 等控制信息 |
| 单次搬运量（示意） | 14,336 B | 64 B |
| Payload 是否过 UB | 是 | 否 |
| MTE 是否是主要瓶颈 | 通常是 | 通常不是，但可能限制下发速率 |

假设一个 Token 是 BF16、hidden size = 7168，则 7168 × 2 B = 14,336 B。WQE 按 64 B 估计（仅用于举例）：

```text
14,336 B / 64 B = 224 倍
```

即两种方式下 MTE3 的搬运字节量相差两个数量级。但要注意三个限定：

- 这是**搬运字节量**的比值，不是实际通信性能提升倍数。
- 64 B 是 SQ 的最小槽位粒度（WQEBB），真实 SQE 可能更大，例如 HCCL MoE 算子里的 Write-with-Notify SQE 是 96 B、占两个槽位。实际比值要按任务格式修正。
- 没有计入量化、打包、Doorbell、CQE 轮询等额外开销。

即便如此，「MTE3 从搬 14 KB 变成搬 64 B」这个量级变化，足以改变 kernel 的瓶颈位置：从 MTE 带宽受限，变成下发速率受限。

## 五、为什么要多个 Jetty

可以把 Jetty 理解成独立的通信任务上下文。多个 Jetty 的主要价值：

- 减少多个 AIV 竞争同一个 SQ 的开销。
- 提高通信任务的并行下发能力。
- 增加可同时在途的通信请求数量。
- 减少不同目标 Rank 之间的队列阻塞。
- 在硬件资源允许时，提高通信引擎和物理链路的利用率。

但必须把几个「数量」分清楚：

| 概念 | 实际含义 |
|---|---|
| Jetty 数量 | 逻辑通信端点资源数量 |
| SQ Depth | 队列能够容纳的任务数量 |
| UDMA Engine 数量 | 物理通信执行资源数量 |
| Port 数量 | 物理互联端口数量 |
| Transport Channel 数量 | 底层传输通道资源数量 |

增加 Jetty 数量，并不等于增加物理 UDMA 引擎数量，也不等于增加物理通信端口数量。多个 Jetty 可以共享底层通信硬件，具体调度和映射取决于芯片架构与通信库配置。

几条已公开的硬约束（细节见 [../../UB/Jetty.md](../../UB/Jetty.md)）：

- Ascend 950 的 STARS2.0 调度器提供 **64 个 UB jetty**，Jetty 是有限资源，建多了会挤压其他通信域。
- 单 Jetty 的 SQ 深度可配，会向上取整到 2 的幂，并有设备上限。
- HCCL 的多通道算法（MeshMultiJetty / NHRMultiJetty）通过 `HCCL_UB_MULTI_CHANNEL_NUM` 控制，只有单 Jetty 是带宽瓶颈时才有收益。
- 省资源的反向做法是共享 Jetty（`IS_SHARED_QUEUE` + 相同 `SHARED_QUEUE_TAG`），代价是共享同一个 Jetty 的通道不支持并发使用。

## 六、Dispatch 与 Combine 的阶段化性能模型

分析 DeepEP-Ascend 这类 EP 通信实现时，建议把一次通信拆成六个阶段，各自盯不同的指标：

| 阶段 | 性能指标 | 主要硬件瓶颈 |
|---|---|---|
| Issue | WQE/s、Issue latency | AIV Scalar、UB、MTE3、Doorbell |
| Queue | SQ occupancy、队列等待 | Jetty、SQ Depth、调度 |
| Transfer | URMA latency、Payload GB/s | UDMA、HBM、UB Bus |
| Completion | CQE polling latency | CQ、AIV、完成处理 |
| Drain | 最后一次提交至全部完成的时间 | 在途请求、链路、慢 Rank |
| Epilogue | 数据重排及聚合耗时 | AIV、MTE2、Vector、MTE3 |

这样拆的好处是：当你看到「Dispatch 带宽只有几十 GB/s」时，能快速判断是 Issue 卡住、Queue 满了，还是 Transfer 真的慢。把六个阶段混成一个数，就只能猜。

DeepEP-Ascend 公开仓库明确采用 HCCL/HCOMM、UBMEM 与 URMA 通信栈，因此它的 Jetty 用法值得单独核对。但具体创建多少 Jetty、AIV 如何分配到 Jetty，需要看它的 kernel 与通信初始化代码，**不能仅从 URMA 通用架构推断**。

### 6.1 小 Token 通信可能受 Issue 能力限制

假设每个 WQE 对应一个 Token，则 UDMA 的有效带宽上限近似为：

```text
B_issue = R_WQE * S_payload
```

其中 `R_WQE` 是每秒能提交的 WQE 数量，`S_payload` 是每个 WQE 对应的有效数据大小。

取 `R_WQE = 10 M WQE/s`，并假设物理有效带宽上限为 1000 GB/s：

| 单个 WQE 的有效 Payload | Issue 对应带宽 | 占 1000 GB/s 链路的比例 |
|---|---|---|
| 1 KiB | 10.24 GB/s | 1.0% |
| 7 KiB | 71.68 GB/s | 7.2% |
| 14 KiB | 143.36 GB/s | 14.3% |
| 56 KiB | 573.44 GB/s | 57.3% |

这解释了一个常见现象：明明 UDMA 和物理链路的理论带宽很高，实际 Dispatch 带宽却很低。**瓶颈可能不是硬件搬不动，而是 AIV 来不及向 Jetty 提交足够多的任务。**

注意这只是量级估算：没考虑 HBM、协议头、队列等待与同步开销，`R_WQE` 本身也随实现和硬件变化。

对应的优化方向：

- WQE 批量提交，减少 Doorbell 次数。
- Doorbell 合并，一次通知对应多个任务。
- 多 Jetty 并行，摊开队列竞争。
- Token 聚合，把小包攒成大包。
- Scatter-Gather，一个 WQE 描述多段数据。
- 减少重复发送（去重、按 Rank 聚合）。

## 七、三种通信方式对比

CANN 的 `MoeDistributeDispatch` / `MoeDistributeCombine` 系列算子接口里，`commType` 直接把「用哪个通信方案」暴露成了参数：`0` 表示 AICPU-SDMA，`1` 表示 CCU，`2` 表示 URMA；当前版本仅支持 `2`，且 Ascend 950DT 系列产品仅支持 URMA 通信。这说明在这个算子里，URMA 是当前唯一可选的通信方案，SDMA 与 CCU 只保留了枚举位。

把三种方式放在一起看：

| 对比项 | AIV MTE 直搬 | AIV 直驱 RDMA | AIV 直驱 URMA |
|---|---|---|---|
| 控制方 | AIV | AIV | AIV |
| Payload 搬运方 | MTE | RDMA Engine | UDMA Engine |
| Payload 经过 AIV UB | 是 | 通常不需要 | 通常不需要 |
| WQE 可能经过 UB | 不涉及此类 WQE | 是 | 是 |
| 通信队列 | MTE 指令流水 | QP | Jetty |
| Doorbell | 不涉及此类 Doorbell | 是 | 是 |
| 异步完成管理 | MTE 同步机制 | CQE | CQE |
| 主要优化方向 | UB 复用、MTE 流水 | WQE、QP、网络 | WQE、Jetty、传输通道 |
| AIV 资源消耗 | 搬运完整数据 | 主要为通信控制 | 主要为通信控制 |

三种方式的共同点是 AIV 都是控制方；差别只在 Payload 由谁搬、经过哪条队列、完成怎么回收。

## 八、平台边界与常见误解

**1. 「950 上是 Jetty，A2/A3 上也是 Jetty。」**
不是。Jetty 是 UB 时代的对象。Atlas A2/A3 上对应的是 SDMA（片内）和 RoCE QP（跨机），没有 Jetty 这一层。在 A2/A3 上找 Jetty，是找错层了。

**2. 「URMA 就是 UDMA。」**
不是。URMA 是机制与接口，UDMA 是执行引擎。公开材料里这两个词经常混用，读文档时要按上下文区分「协议 / 引擎 / 库 / 具体 DMA 操作」。

**3. 「Jetty 越多，带宽越高。」**
只有单 Jetty 是瓶颈时才有收益。单 Jetty 已经打满链路，加 Jetty 只是浪费有限的硬件资源（950 上总共 64 个）。

**4. 「上了 Jetty + UDMA，AIV 就完全退出通信了。」**
没有。AIV 仍然要构造 WQE、提交、轮询 CQE，量化、打包、聚合这些前后处理也还在 AIV 上。MTE3 也仍可能因为搬 WQE 而成为下发瓶颈。

**5. 「224 倍说明通信快了 224 倍。」**
不是。那是 MTE3 搬运字节量的比值，不是端到端性能比值。端到端还受链路、UDMA、同步、Drain 等限制。

**6. 「小包走 UDMA 一定比 MTE 快。」**
不一定。小包时约束往往在 Issue：每个 Token 一个 WQE，`R_WQE` 上不去，带宽就上不去。这也是为什么小包场景下仍有人选择 MTE 直写。

## 参考资料

- [Ascend 950 NPU 架构白皮书](https://public-download.obs.cn-east-2.myhuaweicloud.com/ascend/%E6%98%87%E8%85%BE950%20NPU%E6%9E%B6%E6%9E%84%E7%99%BD%E7%9A%AE%E4%B9%A6.pdf)，URMA 与 STARS2.0 调度器章节
- [CANN：MoE Dispatch & Combine 通算融合算子](https://www.hiascend.com/developer/techArticles/20250726-1)
- [Serving Large Language Models on Huawei CloudMatrix384](https://arxiv.org/html/2506.12708v3)
- [昇腾开源 SHMEM 通信库](https://github.com/hicann/shmem)
- [DeepEP-Ascend](https://github.com/deepseek-ai/DeepEP-Ascend)

本机 CANN 知识库中的对应材料：

- 950 调度器与互连（64 UB jetty、URMA 的 Jetty SQE/CQE、RTP 4 端口 / CTP 9 端口）—— `knowledge/common/platforms/concepts/ascend950_scheduler_interconnect.md`
- 通信引擎对比与 AIV 引擎约束 —— `cann-docs-raw/hcomm/docs/zh/comm_op_dev_guide/prog_models_concepts/comm_engine.md`
- AIV 直驱 URMA 的建链约束 —— `cann-docs-raw/hcomm/docs/zh/api_ref/comm_opdev/control_plane_api/comms_domain_resource_mgmt/HcclChannelAcquire.md`
- AIV 引擎支持的通信场景（URMA 与 UB Memory）—— `cann-docs-raw/hcomm/docs/zh/api_ref/comm_mgr_c/HcclCommSymWinRegister.md`
- `commType` 的三种方案取值 —— `cann-docs-raw/ops-transformer/mc2/moe_distribute_dispatch_setup/docs/aclnnMoeDistributeDispatchSetup.md`
- AIV 作为发起引擎、基于 hcomm + UDMA 的通信接口 —— `cann-docs-raw/ops-transformer/mc2/common/op_kernel/apace/README.md`
- MTE 与 URMA 两种通信拓扑（`topoType`）—— `cann-docs-raw/ops-transformer/mc2/mega_moe/README.md`

## 证据等级

- **官方文档明确记载**：950 的 64 个 UB jetty 与 URMA 的 Jetty SQE/CQE 语义、RTP / CTP 端口数、AIV 直驱 URMA、AIV 引擎仅支持 URMA 与 UB Memory 场景、`commType` 的三种取值、多通道与共享 Jetty 的生效条件。均可在本机 CANN 知识库归档文档中查到。
- **源码与论文观察**：SQE 96 B、WQEBB 64 B 槽位、SHMEM UDMA 的 UB staging 做法、MoE 状态标志位。属实现细节，不保证跨版本稳定。
- **工程推断**：物流类比、224 倍搬运量比值、Issue 受限的带宽表、六阶段拆解、优化方向排序。这些是分析框架与量级估算，不是实测结论。

---

*路径 A（AIV MTE 直搬）见 [aiv_mte.md](aiv_mte.md)；Jetty 自身的对象模型与排错见 [../../UB/Jetty.md](../../UB/Jetty.md)。*
