# vLLM-Ascend SFA C8 量化原理与代码流程汇总

> 本文基于当前 `vllm-ascend` 源码，集中整理 DSA（Dynamic Sparse Attention）模型在 vLLM-Ascend 上的 C8（8-bit KV Cache 量化）相关特性：
>
> - `enable_sparse_sfa_c8` / `enable_sparse_li_c8` / `c8_enable_reshape_optim` 三个开关的原理、代码架构与相关性；
> - C8 与 fp8 block 量化的具体含义；
> - KV Cache 量化是静态还是动态（MLA 静态 vs SFA 动态）；
> - 开启 C8 后 attention 计算是否必然用量化算子；
> - 代码调用链与典型使用场景。

## 1. 核心结论

```text
C8 = 8-bit KV Cache 量化（A2/A3 用 int8，A5 用 fp8_e4m3fn）
目的：显存减半 + KV 访存带宽收益（decode 逐 token 读全量 KV 时尤其显著）
```

- **DSA 模型有两套 KV 数据**：SFA 主 KV Cache（供稀疏 Flash Attention 算最终注意力）和 LightningIndexer (LI) 缓存（选出每个 query 关注的 top-k token）。C8 特性分别量化这两套缓存。
- `enable_sparse_sfa_c8`：SFA 主 KV Cache 量化并**打包**成单一张量（量化 `k_nope` + bf16 `k_rope` + 逐块 scale 元数据）；
- `enable_sparse_li_c8`：LightningIndexer 的 K/Q 量化（int8/fp8 + per-token scale）；
- `c8_enable_reshape_optim`：LI C8 缓存写入改用 `StoreKVBlock` 算子加速，**强依赖** `enable_sparse_li_c8`；
- **SFA 用动态量化**（运行时归约 scale，per-block/per-token），**MLA 用静态量化**（固定 per-layer descale 常数）；
- **开启 C8 后 attention 必然走量化算子** `npu_kv_quant_sparse_flash_attention`——路由依据是缓存张量的 dtype，不存在"只存储量化、计算照旧"的中间态。

## 2. 概念：C8 与 fp8 block 量化

### 2.1 C8 是什么

C8 = "Cache 8-bit"。dtype 由硬件决定（`vllm_ascend/attention/sfa_v1.py:610-616`、`worker/model_runner_v1.py:381-387`）：

```python
if get_ascend_device_type() == AscendDeviceType.A5:
    self.c8_k_cache_dtype = torch.float8_e4m3fn   # Ascend 950
    self.c8_k_scale_cache_dtype = torch.float32
else:
    self.c8_k_cache_dtype = torch.int8            # A2/A3
    self.c8_k_scale_cache_dtype = torch.float16
```

### 2.2 fp8 block 量化

block 量化 = 量化粒度是"块"而非整个张量。SFA 主 KV 对 `k_nope` 做**逐 128 元素一块**的量化，每块独立一个 scale（`sfa_v1.py:2027-2032`）：

```python
k_nope, knope_scale = torch_npu.npu_dynamic_block_quant(
    k_nope.contiguous().view(-1, 1, kv_lora_rank),  # [token, 1, kv_lora_rank]
    dst_type=dst_type,               # fp8_e4m3fn (A5) 或 int8
    row_block_size=1,                # token 方向块高 = 1
    col_block_size=tile_size,        # SFA_QSFA_TILE_SIZE = 128
)
```

以 DeepSeek `kv_lora_rank=512` 为例：一个 token 的 K 被切成 512/128 = 4 块，每块输出 1 个 fp32 scale。

**为什么必须 block 量化（尤其 fp8）**：fp8 e4m3fn 只有 3 位尾数，动态范围窄。若只用一个 per-token scale 覆盖整条 512 维 K 向量，局部幅值差异会顾此失彼（大段溢出、小段 underflow）。逐 128 块给独立 scale 可捕捉局部幅值。配套还有 fp8 专用的 clip 因子 `sfa_qsfa_k_nope_clip_alpha`（`sfa_v1.py:826-830`，初值 1.0）。

### 2.3 packed 缓存布局

量化结果不是独立存 scale，而是和 K 一起打包进**同一个 head 维**（`get_sfa_qsfa_packed_head_dim`，`vllm_ascend/attention/utils.py:33-44`）：

```text
packed_head_dim = kv_lora_rank（量化后 8bit）
                + qk_rope_head_dim × 2B（bf16 rope）
                + (kv_lora_rank // 128) × 4B（fp32 scale，每块一个）
```

写入由 `custom_kv_rmsnorm_rope`（`sfa_v1.py:2010-2046`）产出三段：`k_rope`（bf16 字节）、`k_nope`（int8/fp8）、`knope_scale`（fp32 scale），调用方再 `torch.cat` 成一行、scatter 进单个缓存张量（`_maybe_store_kvcache_for_c8_n_dsacp`，`sfa_v1.py:1604-1619`）。

## 3. 三个开关的配置与约束

### 3.1 配置定义（`vllm_ascend/ascend_config.py:252-262`）

```python
use_sparse = model_uses_sfa_sparse(model_config)          # DSA 模型检测
self.enable_sparse_sfa_c8 = get("enable_sparse_sfa_c8", False) and use_sparse
self.enable_sparse_li_c8  = get("enable_sparse_li_c8",  False) and use_sparse
self.c8_enable_reshape_optim = self.enable_sparse_li_c8 and get("c8_enable_reshape_optim", False)
```

DSA 模型检测 `model_uses_sfa_sparse`（`vllm_ascend/utils.py:111-119`）：`hf_text_config` 有 `index_topk` 且无 `compress_ratios`（即 DeepSeek V3.2、GLM5 这类，区别于 DeepSeek-V4 的 `compress_ratios` DSA 变体）。

### 3.2 硬约束

- 两个 C8 开关都要求 `use_sparse` 为真，非 DSA 模型设了也无效；
- `c8_enable_reshape_optim` 是 `enable_sparse_li_c8` 的充分条件式依赖（AND 关系），LI C8 不开则 reshape 优化自动失效；
- **LI C8 有"层过滤"**：`_parse_sparse_li_c8_layers_from_quant_config`（`ascend_config.py:411-435`）从 `quant_config.quant_description` 中找 `.indexer.quant_type` / `.indexer.wq_b_weight` 为 `INT8_DYNAMIC` / `W8A8_MXFP8` 的层，只对量化配置中标记了 C8 的 indexer 层生效；无该量化配置则全部 indexer 层生效（`is_sparse_li_c8_layer`，`ascend_config.py:437-454`）。

## 4. 静态 vs 动态量化

### 4.1 MLA（`vllm_ascend/attention/mla_v1.py`）：静态，per-layer 固定 scale

scale 是权重加载时算好的固定常数，来自 W8A8 量化配置（`_process_weights_for_fused_fa_quant`，`mla_v1.py:973-995`）：

```python
layer = self.vllm_config.compilation_config.static_forward_context[self.layer_name]
self.fak_descale_float       = layer.fak_descale_float
self.quant_kscale            = layer.quant_kscale
self.fak_descale_reciprocal  = layer.fak_descale_reciprocal
```

- 写缓存（`exec_kv_decode`，`mla_v1.py:1311-1323`）：用固定 `c_kv_scale = fak_descale_reciprocal` 缩放成 int8/fp8 存入 cache；
- 读缓存反量化（`mla_v1.py:839-841`）：attention op 用固定 `dequant_scale_key = dequant_scale_value = fak_descale_float`。

即：**量化 scale 在权重处理阶段定死，推理零运行时开销，per-layer 一个标量**。

### 4.2 SFA（`vllm_ascend/attention/sfa_v1.py`）：动态，per-block/per-token 运行时算 scale

- SFA 主 KV：`npu_dynamic_block_quant`（`sfa_v1.py:2027-2032`）——每 128 元素一块，**根据当前 KV 实际数值现算 scale**，随后打进缓存（`quant_scale_repo_mode=1`），attention 核读 scale repo 反量化；
- LI 索引：`npu_dynamic_quant(k_li.view(-1, self.head_dim))`（`sfa_v1.py:1364`、`1439`）——**per-token** 动态 scale。

### 4.3 对比

| | MLA FA quant（静态） | SFA C8（动态） |
|---|---|---|
| scale 来源 | 离线算好，固定常数 | 运行时从 KV 数据归约 |
| 粒度 | per-layer 一个标量 | per-128-block（主 KV）/ per-token（LI） |
| 写缓存 | `c_kv_scale`（常数） | `npu_dynamic_block_quant` / `npu_dynamic_quant` |
| 反量化 | `fak_descale_float`（常数） | 缓存里的 scale repo |
| 精度 | 受固定 scale 限制 | 局部幅度自适应，精度高 |
| 开销 | 0（无运行时归约） | 每次多一次 scale 计算 |

**为什么并存**：
- SFA 用动态是"必须的"——主 KV 走 fp8（A5），fp8 动态范围窄，逐块动态 scale 才能保精度；DSA 长上下文场景精度敏感。代价是每次 forward 多一次归约。
- MLA 用静态是"够用 + 零开销"——复用 W8A8 权重量化已有的 descale 因子，无需运行时归约，实现简单。

> 注意：MLA 里也有 `npu_dynamic_quant`（`mla_v1.py:1617`，A5 上对 `decode_ql_nope`），但那是 **query 激活量化**，不是 KV Cache 的 scale。KV Cache 的 scale 在 MLA 中始终静态。

## 5. 量化后 attention 计算路径

### 5.1 dtype-based 路由（`vllm_ascend/device/device_op.py:543-551`）

```python
use_kv_quant_sparse_attention = kv.dtype in (torch.int8, torch.float8_e4m3fn, torch.float8_e5m2)
if use_kv_quant_sparse_attention:
    # → npu_kv_quant_sparse_flash_attention（量化算子）
else:
    # → npu_sparse_flash_attention（BF16/FP16 算子）
```

**因果链**：C8 开启 → 写缓存量化 → `kv_cache[0].dtype` 是 int8/fp8 → 路由必然命中量化分支。注释明确 "The kv-quant sparse attention op only accepts packed quantized KV"，即 **存储量化与计算量化强绑定**，不存在中间态。dtype-based 而非 flag-based 是刻意设计，让测试/回退路径可喂 BF16 cache 跑通同一段代码。

### 5.2 量化算子的精确边界（`device_op.py:608-629`）

```python
query = torch.cat([ql_nope, q_pe], dim=-1).contiguous()   # Q 全精度（bf16/fp16）
torch.ops._C_ascend.npu_kv_quant_sparse_flash_attention(
    query=query, key=kv, value=kv,
    quant_scale_repo_mode=1,   # scale 从打包缓存内的 repo 读取
    key_quant_mode=2, value_quant_mode=2,
    tile_size=128, ...)
```

| 环节 | 是否量化 | 说明 |
|---|---|---|
| K/V 存储 | ✅ int8/fp8 | 主要目的，显存减半 |
| K/V 加载 | ✅ 量化域 | 按 128 tile 从 cache 读 int8 |
| K/V 反量化 | 算子内 | 用 scale repo 反量化成 bf16 |
| QK^T 点积 | ❌ 全精度 | Q 即时算、全精度，K 反量化后做点积 |

收益本质是**访存带宽 + 显存**，而非 int8 matmul 加速。

### 5.3 LI 索引：真·量化域计算（`device_op.py:479-494`）

```python
topk_indices = torch.ops._C_ascend.npu_lightning_indexer_quant(
    query=q_li,                     # int8（Hadamard + dynamic_quant）
    key=kv_cache[indexer_k_idx],    # int8 缓存
    query_dequant_scale=q_li_scale,
    key_dequant_scale=kv_cache[indexer_scale_idx], ...)
```

q_li 与 K 均以 int8 加载、带各自 scale 参与 top-k 匹配打分——这是**真正的 int8 域量化计算**。不开 C8 时走 `npu_lightning_indexer`（BF16）。

### 5.4 prefill 与 decode

prefill 与 decode 共用同一 `_execute_sparse_flash_attention_process` 路径（`sfa_v1.py:1975`）。prefill 把新算的 KV 也量化后 scatter 进 cache，再统一从 cache 读。因此只要 cache dtype 是 int8/fp8，**prefill 同样走量化算子**，没有独立的 BF16 prefill 分支。

## 6. 代码架构（调用链）

```
① 配置层 ascend_config.py
   use_sparse 检测 → 三个开关解析 → LI 层过滤 (is_sparse_li_c8_layer)
        │
② 缓存 spec / 分配层
   worker/model_runner_v1.py L4698-4756 构建 kv_cache_spec：
   ├─ MLAAttention（SFA 主）：enable_sparse_sfa_c8 → AscendMLAAttentionSpec
   │     head_size = get_sfa_qsfa_packed_head_dim(...), dtype = c8_k_cache_dtype
   └─ DeepseekV32IndexerCache：is_sparse_li_c8_layer → AscendSFAIndexerCacheSpec
         dtype=c8_k_cache_dtype, scale_dim=1, scale_dtype=c8_k_scale_cache_dtype
   ├─ L3989 _allocate_sparse_c8_indexer_tensors：k + scale 从同一块 int8 内存切出
   └─ L4410-4418：SFA C8 时 v_cache=None（无独立 V 缓存，packed）
        │
③ 注意力实现层 sfa_v1.py
   AscendSFAImpl 持 enable_sparse_sfa_c8 / enable_sparse_li_c8
   ├─ exec_kv → custom_kv_rmsnorm_rope（打包量化 + concat）
   ├─ indexer 处理 → npu_dynamic_quant（K/Q C8）
   └─ forward → _maybe_store_kvcache_for_c8_n_dsacp / store_kv_block
        │
④ 设备算子层 device_op.py
   npu_kv_quant_sparse_flash_attention（C8 SFA 主注意力） L608
   npu_lightning_indexer_quant（C8 索引匹配） L479
   indexer_select_post_process（量化 scale 分支） L474-481 / L1666-1673
        │
⑤ 并行/外围层
   attention/context_parallel/sfa_cp.py L470：DCP 下 C8 打包缓存整块 all-gather
   attention/sfa_kv_offload.py L167：SFA C8 主缓存不支持 KV offload（仅 LI C8 支持）
   xlite/xlite.py L751：TODO 兼容 SFA C8
```

## 7. KV Cache 布局与三个开关的关系

三个开关共同改变 KV Cache tuple 布局（`sfa_v1.py:643-661`）：

```text
默认:      kv_cache[0]=k_nope  [1]=k_pe    [2]=indexer_k  [3]=indexer_scale
SFA C8:    kv_cache[0]=packed  [1]=indexer_k  [2]=indexer_scale  [3]=(unused)
两 C8 全开: kv_cache[0]=packed  [1]=indexer_k  [2]=indexer_scale
```

因此代码用 `kv_cache_indexer_k_idx` / `kv_cache_indexer_scale_idx` 按 `enable_sparse_sfa_c8` 动态偏移，并大量使用 `len(kv_cache) == (3 if enable_sparse_sfa_c8 else 4)` 断言保证布局一致。

| 维度 | `enable_sparse_sfa_c8` | `enable_sparse_li_c8` | `c8_enable_reshape_optim` |
|---|---|---|---|
| 作用对象 | SFA 主 KV Cache | LightningIndexer 缓存 | LI C8 的写入方式 |
| 前置条件 | `use_sparse`（DSA 模型） | `use_sparse` + 量化层过滤 | `enable_sparse_li_c8` |
| 硬件 dtype | A5=fp8 / 其余=int8 | 同左 | 无关 |
| 缓存布局变化 | 2 张量→1 packed | 1→2 (k, scale) | 无布局变化，仅写法 |
| 是否互相独立 | ✅ 独立 | ✅ 独立 | ❌ 依赖 LI C8 |

## 8. 使用场景

面向 **W4A8C8 这类 C8 量化模型**（如 `GLM-5.2-w4a8c8`，见 `docs/tutorials/models/GLM5.2.md`）。典型配置：

```json
{"enable_dsa_cp": true, "enable_sparse_li_c8": true, "enable_balance_scheduling": true}
```

推荐逻辑（`GLM5.2.md:187`）：`enable_sparse_li_c8` 建议常开加速 LI 稀疏注意力；显存不足（长序列）再开 `enable_sparse_sfa_c8`；PD 分离的 P 节点可配 `c8_enable_reshape_optim`（`GLM5.2.md:479`）。

三个开关构成"主 KV 量化（SFA C8）→ 索引量化（LI C8）→ 索引写入加速（reshape optim）"的递进优化链。前两个互相独立、可组合出多种 KV 布局；第三个是第二个的可选加速，与 LI C8 强绑定。另注意与 `enable_kv_nz` 互斥（`ascend_config.py:245` 明确禁止 KV NZ 与稀疏模型同开）。
