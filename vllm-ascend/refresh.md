# vllm-ascend `refresh` 配置说明

基于当前仓库 commit `aa48bd6879` 分析，`additional_config.refresh` 的核心作用是：

> 强制 vLLM-Ascend 重新创建进程级的 `AscendConfig` 单例。它不是模型权重刷新开关，也不会直接更新模型参数、KV Cache 或 NPU Graph。

## 1. 配置定义

官方文档将其定义为：

| 字段 | 类型 | 默认值 | 作用 |
|---|---:|---:|---|
| `refresh` | `bool` | `false` | 刷新全局 Ascend 配置，通常用于 RLHF 或 UT/E2E |

对应文档见 [additional_config.md](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/docs/source/user_guide/configuration/additional_config.md:72)，也可参考[官方在线文档](https://docs.vllm.ai/projects/ascend/en/latest/user_guide/configuration/additional_config.html)。

使用方式：

```python
llm = LLM(
    model=model_path,
    additional_config={
        "enable_fused_mc2": 0,
        "multistream_overlap_shared_expert": True,
        "refresh": True,
    },
)
```

注意必须传 Python 布尔值 `True`，不要传字符串 `"true"` 或 `"false"`。当前实现没有严格类型校验，而是直接使用 Python truthiness；字符串 `"false"` 实际上也会被当成真值。

## 2. 核心实现

入口位于 [ascend_config.py](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/vllm_ascend/ascend_config.py:1088)：

```python
def init_ascend_config(vllm_config):
    additional_config = (
        vllm_config.additional_config
        if vllm_config.additional_config is not None
        else {}
    )
    refresh = additional_config.get("refresh", False)

    global _ASCEND_CONFIG

    if (
        _ASCEND_CONFIG is not None
        and not refresh
        and _is_ascend_config_initialized(_ASCEND_CONFIG)
        and _ASCEND_CONFIG.vllm_config is vllm_config
    ):
        return _ASCEND_CONFIG

    new_config = AscendConfig(vllm_config)
    _ASCEND_CONFIG = new_config
    return new_config
```

其行为可归纳为：

| 当前状态 | `refresh` | `VllmConfig` 对象 | 结果 |
|---|---:|---|---|
| 尚未初始化 | 任意 | 任意 | 创建 `AscendConfig` |
| 已初始化 | `False` | 同一个对象 | 复用旧单例 |
| 已初始化 | `False` | 新对象 | 重新创建 |
| 已初始化 | `True` | 同一个或新对象 | 强制重新创建 |

因此，`refresh=True` 真正解决的是：

> 同一个 `VllmConfig` Python 对象被修改后，再次调用 `init_ascend_config()` 时，不能继续复用旧的解析结果。

## 3. 刷新的具体内容

`AscendConfig` 构造时会重新读取整个 `additional_config`，而不只是某个字段。入口见 [AscendConfig.__init__](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/vllm_ascend/ascend_config.py:32)。

重新解析的内容包括：

- Graph/编译配置
- Fine-grained TP 配置
- EPLB 配置
- 调度器扩展配置
- FlashComm1
- Fused MC2
- MLAPO
- NZ 权重格式
- Shared Expert DP
- CPU Binding
- KV Cache 相关配置
- Dump、日志和其他 Ascend 特有配置

流程大致如下：

```text
VllmConfig.additional_config
          │
          ▼
读取 refresh
          │
          ├─ refresh=False 且同一 VllmConfig
          │       └─ 返回已有 _ASCEND_CONFIG
          │
          └─ refresh=True / 使用新的 VllmConfig
                  │
                  ▼
            AscendConfig(vllm_config)
                  │
                  ├─ 重新解析所有 Ascend 配置
                  ├─ 执行配置校验
                  ├─ 读取部分环境变量回退值
                  └─ 替换全局 _ASCEND_CONFIG
```

此外，`refresh` 还被 [enable_sp()](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/vllm_ascend/utils.py:814) 使用：

```python
refresh = additional_config.get("refresh", False)

if _ENABLE_SP is None or refresh:
    # 重新计算 FlashComm1 / sequence-parallel 开关
```

也就是说，开启 `refresh` 时，除了 `_ASCEND_CONFIG`，`_ENABLE_SP` 这个独立缓存也会重新计算。

## 4. 初始化调用位置

Ascend 配置主要在两个阶段初始化。

首先是平台配置检查阶段：

[NPUPlatform.check_and_update_config](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/vllm_ascend/platform.py:346)

```python
_fix_incompatible_config(vllm_config)
ascend_config = init_ascend_config(vllm_config)
```

其次是每个 NPU Worker 初始化阶段：

[NPUWorker.__init__](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/vllm_ascend/worker/worker.py:129)

```python
init_ascend_config(vllm_config)
```

需要注意，`_ASCEND_CONFIG` 是 Python 进程内全局变量，不是跨进程全局变量：

- Driver 有自己的 `_ASCEND_CONFIG`
- 每个 MP/Ray Worker 也有自己的 `_ASCEND_CONFIG`
- `refresh=True` 不会向其他进程广播配置
- 各 Worker 必须收到带有相同 `additional_config` 的 `VllmConfig`

## 5. 为什么 RL 场景需要它

RL/RLHF 场景和普通在线推理的生命周期不同。

普通推理通常是：

```text
启动进程 → 创建一次 VllmConfig → 创建模型 → 持续推理
```

RL 场景则可能是：

```text
启动长期存活的 Ray/训练 Worker
    │
    ├─ 导入并初始化 vLLM-Ascend
    ├─ 构造或探测一次默认 VllmConfig
    ├─ 修改同一个 VllmConfig
    ├─ 创建 rollout engine
    ├─ 多轮生成、训练、更新权重
    └─ 复用/重建 rollout engine
```

在以下情况下容易出现旧配置污染：

1. RL 框架在同一进程内复用 Worker。
2. `AscendConfig` 已被默认配置或前一个推理引擎初始化。
3. 框架随后修改同一个 `VllmConfig.additional_config`。
4. 再次初始化 rollout engine。
5. 如果不刷新，就可能继续使用之前缓存的 `AscendConfig`。

例如：

```python
vllm_config.additional_config = {
    "enable_fused_mc2": 0,
}

init_ascend_config(vllm_config)

# RL 框架之后修改同一个对象
vllm_config.additional_config = {
    "enable_fused_mc2": 1,
}

# refresh=False 时，同一个 VllmConfig 会命中缓存
ascend_config = init_ascend_config(vllm_config)
# 仍可能是旧的 enable_fused_mc2=0
```

使用：

```python
vllm_config.additional_config = {
    "enable_fused_mc2": 1,
    "refresh": True,
}

ascend_config = init_ascend_config(vllm_config)
# 强制创建新的 AscendConfig
```

这个字段最初就是在修复 RLHF 配置复用问题时加入的，提交说明明确写了“add refresh config for rlhf case”。

## 6. 当前版本中是否仍然必须开启

当前实现已经增加了对象身份判断：

```python
_ASCEND_CONFIG.vllm_config is vllm_config
```

因此分为两种情况。

### 创建了新的 `VllmConfig`

```python
config1 = VllmConfig(...)
config2 = VllmConfig(...)
```

当前版本会自动发现这是不同对象，并重新创建 `AscendConfig`。这种情况下通常不再需要 `refresh=True`。

对应单元测试见 [test_ascend_config.py](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/tests/ut/test_ascend_config.py:481)。

### 原地修改同一个 `VllmConfig`

```python
config.additional_config["enable_fused_mc2"] = 1
init_ascend_config(config)
```

因为仍然是同一个对象，只有 `refresh=True` 才会跳过缓存。

所以在当前版本中，`refresh` 更准确的使用条件是：

> RL 框架会不会在同一个 Python 进程中，原地修改并重复使用同一个 `VllmConfig` 对象。

## 7. 它与 RL 权重更新没有直接关系

RL 每轮通常包含：

```text
rollout 生成
    ↓
训练模型计算梯度
    ↓
更新训练模型权重
    ↓
暂停 vLLM 生成
    ↓
把新权重传给 vLLM
    ↓
恢复生成
```

vLLM-Ascend 示例中的权重流程是：

```text
pause_generation
    ↓
start_weight_update
    ↓
update_weights / send_weights
    ↓
finish_weight_update
    ↓
resume_generation
```

HCCL 示例见 [rlhf_http_hccl.py](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/examples/rl/rlhf_http_hccl.py:220)，NPU IPC 示例见 [rlhf_http_npu_ipc.py](C:/Users/liurong/.codex/worktrees/3279/vllm-ascend/examples/rl/rlhf_http_npu_ipc.py:159)。

`refresh` 不参与上述权重传输：

| 操作 | `refresh` 是否负责 |
|---|---:|
| 重新解析 Ascend 配置 | 是 |
| 替换 `_ASCEND_CONFIG` | 是 |
| 重新计算 `_ENABLE_SP` | 是 |
| 更新模型 Parameter 内容 | 否 |
| 传输训练权重到 rollout engine | 否 |
| 清空或更新 KV Cache | 否 |
| pause/resume 推理 | 否 |
| 重新捕获 NPU Graph | 否 |
| 重建 TP/EP/HCCL 通信组 | 否 |

因此，不应该在每一轮 RL 权重更新前后切换 `refresh`。模型权重更新应继续使用 `update_weights`、HCCL 或 NPU IPC 等接口。

## 8. RL 中推荐用法

对于同进程、Worker 复用或配置对象可能被原地修改的 RL 框架：

```python
rollout_additional_config = {
    # 真正的 Ascend 功能配置
    "enable_fused_mc2": 0,
    "multistream_overlap_shared_expert": False,

    # 防止复用进程中的旧 AscendConfig
    "refresh": True,
}

rollout_engine = LLM(
    model=model_path,
    enable_sleep_mode=True,
    additional_config=rollout_additional_config,
)
```

使用原则：

- 在创建 rollout engine 时设置。
- 所有 Ray/MP Worker 应获得一致配置。
- 不需要在每个 rollout/training step 修改。
- 单纯更新模型权重不需要它。
- 独立启动的 HTTP vLLM Server 通常是全新进程，通常不需要它。
- 当前版本每次使用新的 `VllmConfig` 时会自动刷新，`refresh=True` 更多是兼容旧版本和同对象复用场景。

## 9. 主要限制

`refresh=True` 只替换全局配置对象，不会重建已经创建的运行时组件。

例如以下组件可能已经在构造阶段读取并保存配置：

- Attention backend
- FusedMoE
- Sampler
- 编译器和 NPU Graph
- TP/EP 通信组
- Scheduler
- KV Connector

如果模型已经完成初始化，再动态修改：

```python
additional_config["finegrained_tp_config"] = ...
additional_config["enable_fused_mc2"] = 1
```

即使重新调用 `init_ascend_config()`，也不能保证这些现存组件同步改变。涉及通信拓扑、权重布局、算子选择或 Graph 的配置，仍应重建 rollout engine。

最终可以将 `refresh` 理解为：

> 一个解决进程内 Ascend 配置单例缓存污染的初始化期开关；它适用于 RL Worker/测试进程复用，不是 RL 模型权重热更新机制。