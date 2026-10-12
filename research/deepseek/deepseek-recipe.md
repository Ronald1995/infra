# DeepSeek-Recipe 架构分析及与 vLLM/SGLang 生态关系分析

## 1. 背景

随着 DeepSeek V4/V4.1 等模型能力增强，模型服务层面逐渐出现更多与模型强相关的协议需求，包括：

- DeepSeek 特定 Prompt 模板
- Thinking/Reasoning 模式
- Tool Calling 格式
- 多模态输入处理
- Streaming Response 解析
- OpenAI API 兼容

如果每一个推理框架（如 vLLM、SGLang、TensorRT-LLM、自研 Runtime）分别实现这些逻辑，会导致：

1. 多套 DeepSeek 协议实现长期维护；
2. 不同后端之间 Prompt 不一致；
3. 训练和推理阶段可能出现格式偏差；
4. 新模型版本适配成本增加。

因此 DeepSeek 发布了 `deepseek-recipe` 项目，目标是提供一个统一的模型协议适配层。

---

# 2. DeepSeek-Recipe 定位

## 2.1 核心定位

DeepSeek-Recipe 不是推理框架，而是：

> DeepSeek 模型生态中的协议转换和 Prompt 编解码组件。

它负责：

```
API Request
      |
      v
Conversation Format
      |
      v
DeepSeek Prompt Encoding
      |
      v
Inference Backend
      |
      v
Generated Output
      |
      v
API Response
```

其中推理过程由外部系统提供：

- vLLM
- SGLang
- TensorRT-LLM
- 自研 Runtime

DeepSeek-Recipe 不负责：

- 模型执行
- GPU/NPU 调度
- KV Cache
- Batch Scheduler
- Kernel 优化

---

# 3. 整体架构

仓库采用 Rust 核心 + Python Binding 架构。

整体结构：

```
deepseek-recipe

├── deepseek-recipe
│       协议转换
│       Request/Response Parser
│
├── deepseek-recipe-core
│       Conversation 数据结构
│
├── deepseek-recipe-encoding
│       DeepSeek V4/V4.1 Prompt 编码
│
├── deepseek-recipe-image
│       图像输入处理
│
├── deepseek-recipe-python
│       Python API
│
├── server-rs
│       Rust Server Demo
│
├── server-py
│       Python Server Demo
│
└── encoding-decoding-demo
        Prompt 调试工具
```

---

# 4. 核心模块分析

## 4.1 deepseek-recipe-core

### 作用

提供统一中间表示。

不同 API：

```
OpenAI ChatCompletion

Anthropic Messages

Responses API

        |
        v

Conversation
```

统一转换为：

```
Conversation

{
 messages,
 tools,
 images,
 thinking_config
}
```

优势：

后续 Prompt 编码不关心请求来源。

---

# 4.2 deepseek-recipe

## 协议转换层

主要负责：

## Request Conversion

例如：

输入：

```json
{
 "messages": [
   {
     "role": "user",
     "content": "hello"
   }
 ]
}
```

转换：

```
ChatCompletionRequest

        |

Conversation
```


支持：

- Chat Completions
- Messages
- Responses API


---

## Response Parsing

模型输出：

```
<think>
reasoning
</think>

answer
```

解析：

```
Thinking Content

Tool Call

JSON Output

Normal Text
```

转换为：

OpenAI compatible response。

---

# 4.3 deepseek-recipe-encoding

这是最核心模块。

作用：

生成 DeepSeek 专用 Prompt。

流程：

```
Conversation

      |

DeepSeek V4/V4.1 Encoding

      |

Prompt String

      |

Tokenizer

      |

Input IDs
```

例如：

普通模型：

```
User:
hello
Assistant:
```

DeepSeek：

```
<special token>
<User>
hello
<Assistant>
<thinking marker>
```

因此保证：

- DeepSeek chat template 一致；
- thinking token 正确；
- tool token 正确。

---

# 4.4 deepseek-recipe-image

负责 DeepSeek V4.1 多模态输入。


流程：

```
Image URL/Base64

        |

Decode

        |

Preprocess

        |

Image Token
```

主要处理：

- 图片加载；
- OpenCV preprocessing；
- 多模态输入转换。

---

# 4.5 Python Binding

提供：

```python
from deepseek_recipe import (
    ChatCompletionRequest,
    DeepseekV41Encoding
)
```

方便：

- FastAPI 服务；
- Agent Framework；
- Python 推理平台。


---

# 5. DeepSeek-Recipe 与 vLLM 的关系

## 5.1 当前 vLLM 架构

当前 vLLM：

```
OpenAI API

      |

Chat Template

      |

Tokenizer

      |

Model Runner

      |

DeepSeek Model
```

DeepSeek 支持主要通过：

- tokenizer 配置；
- chat template；
- reasoning parser；
- tool parser。

---

## 5.2 为什么 vLLM 当前没有直接依赖 DeepSeek-Recipe？

主要原因：

两者职责存在重叠。


vLLM 已经实现：

```
OpenAI Server

      |

Protocol Parser

      |

Chat Template

      |

Reasoning Parser

      |

Tool Parser
```

而 DeepSeek-Recipe：

```
API Protocol

      |

Conversation

      |

Prompt Encoder

      |

Response Parser
```

如果直接引入：

```
vLLM

  |

deepseek-recipe

  |

DeepSeek
```

会产生：

- Prompt 逻辑重复；
- Tool Parser 重复；
- Response Parser 重复；
- Runtime 与协议层耦合。

因此当前更可能保持：

```
vLLM
 |
自己的 DeepSeek Adapter
```

---

# 6. DeepSeek-Recipe 与 SGLang 的关系

SGLang 情况类似。

当前 SGLang：

```
SGLang Server

      |

DeepSeek Parser

      |

DeepSeek Runtime

      |

Model
```

同样：

- 自己维护 DeepSeek parser；
- 自己维护 serving 逻辑；
- 自己负责 runtime。


因此目前没有明确迹象表明 SGLang 会把 DeepSeek-Recipe 作为内部依赖。

---

# 7. DeepSeek-Recipe 的潜在应用场景

## 7.1 第三方 Serving Gateway

这是最适合场景。

架构：

```
Client

 |

OpenAI API

 |

DeepSeek-Recipe

 |

vLLM/SGLang/TRT

 |

DeepSeek Model
```

优势：

- 后端可替换；
- 协议统一；
- 降低维护成本。


---

# 7.2 DeepSeek 官方服务

类似：

```
DeepSeek API

 |

DeepSeek-Recipe

 |

Inference Backend
```

推理后端可以变化：

- vLLM
- SGLang
- TensorRT
- 自研系统

而 API 行为保持一致。

---

# 7.3 RL 训练场景

这是与当前 RL 系统最相关的方向。

当前 RL：

```
verl/slime

      |

OpenAI API

      |

vLLM
```

存在潜在问题：

训练侧：

```
Conversation

      |

Prompt Template A
```

推理侧：

```
Conversation

      |

Prompt Template B
```

导致：

```
Train / Inference Prompt Mismatch
```

DeepSeek-Recipe 可以作为：

```
RL Dataset

      |

DeepSeek-Recipe

      |

Prompt IDs

      |

vLLM/SGLang
```

保证：

训练和推理使用同一套格式。

---

# 8. 与 DeepSeek V4.1 技术栈的关系

DeepSeek V4.1 技术创新：

- mHC
- Engram
- SWA
- Attention Sink
- MoE

属于：

```
Model Architecture Layer
```

而 DeepSeek-Recipe：

属于：

```
Serving Protocol Layer
```

两者关系：

```
                 User API

                    |

            DeepSeek-Recipe

                    |

              vLLM/SGLang

                    |

          DeepSeek V4.1 Runtime

                    |

        mHC / Engram / SWA / MoE
```

Recipe 不影响模型计算逻辑。

---

# 9. 当前限制

DeepSeek-Recipe 当前未覆盖：

- logprobs/top_logprobs；
- JSON Schema 强约束；
- 多 candidate 输出；
- server-side tool execution。

其中：

## logprobs

对于 RL 非常重要：

```
Policy Model

      |

Token Probability

      |

Reward Calculation
```

因此 RL 系统仍需要：

- vLLM native API；
- SGLang native API；

而不能完全依赖 Recipe。

---

# 10. 对 vLLM-Ascend 的启示

结合 vLLM-Ascend 当前重点：

- DeepSeek V4/V4.1 支持；
- MLA/SFA；
- KV Cache；
- PWAL；
- Weight Update；
- Attention Sink；
- RL Live Update。


这些属于：

```
Runtime / Model Execution Layer
```

不是 Recipe 解决的问题。

因此：

## 短期

不建议 vLLM-Ascend 引入 DeepSeek-Recipe。

原因：

- Runtime 已经有 DeepSeek 支持；
- 引入会增加协议耦合；
- 对 NPU kernel 无帮助。


## 长期

值得关注两个方向：

### 1. RL Prompt Alignment

统一：

```
训练 Prompt

=

推理 Prompt
```

降低 RL 训推偏差。


### 2. DeepSeek 新模型快速适配

未来：

DeepSeek V5/V6

如果增加：

- 新 thinking 格式；
- 新 tool 协议；
- 新 multimodal 格式；

Recipe 可以降低多个 backend 适配成本。

---

# 11. 总结

DeepSeek-Recipe 可以定义为：

> DeepSeek 官方提供的模型协议适配层，用于统一不同 API 格式、Prompt 模板以及输出解析，使不同推理后端能够一致服务 DeepSeek 模型。

当前生态关系：

```
                 Application

                     |

             DeepSeek-Recipe

                     |

       ----------------------------

       |            |             |

      vLLM       SGLang       TensorRT

       |            |             |

       ----------------------------

                 DeepSeek Model
```

判断：

- **短期：vLLM/SGLang 不太可能直接依赖 deepseek-recipe。**
- **中长期：deepseek-recipe 更可能成为 DeepSeek 官方生态标准协议层。**
- **对于 RL 场景，它最大的价值是保证训练和推理 Prompt/Response 格式一致。**

对于 vLLM-Ascend，当前更值得关注的是：

1. DeepSeek Runtime 优化；
2. RL weight update；
3. Prompt/Token 一致性；
4. 未来是否通过 Recipe 简化 DeepSeek 新模型接入。