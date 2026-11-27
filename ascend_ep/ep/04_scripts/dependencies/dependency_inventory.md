# EP 算子依赖盘点（36 个算子）

> 只读调查。本轮未修改 manifest / install_deps.sh / Doctor / SKILL.md，未安装任何包，未执行 Native Build。

## 0. 结论速览

- 算子总数：**36**（registry、smoke_profiles、case catalog 三者一致）
- 仅需 TUNA wheel 即可闭合依赖：**5**
- 需要第三方源码 + A5 构建：**29**
- 算子级 hardware 约束排除 Ascend950（真平台阻塞，装也无效）：**2**
- 依赖发布说明未列出 Ascend950（需装完后实测，**不是**平台阻塞）：**11**
- 仅需系统/CANN 提供：**0**

依赖可获取 ≠ 算子可运行：本表只回答“依赖能否到位”，运行能力见第 9 节。

## 1. 仓库与环境

| 项 | 值 |
| --- | --- |
| 仓库 | `infra` @ `67b5f99`（main，已含本次重构） |
| 算子清单来源 | `operators/registry.yaml` + `operators/smoke_profiles.yaml` + `common/case_spec.py` |
| manifest 条目 | `dependencies/manifest.yaml`：40 个依赖项 |
| 验证主机 | A5 `rack09os2`，容器 `liurong_vllm` |
| Python / 架构 | 3.12.13 / aarch64（CPython） |
| 兼容标签（前 6） | cp312-cp312-linux_aarch64, cp312-cp312-manylinux_2_38_aarch64, cp312-cp312-manylinux_2_37_aarch64, cp312-cp312-manylinux_2_36_aarch64, cp312-cp312-manylinux_2_35_aarch64, cp312-cp312-manylinux_2_34_aarch64 |
| CANN | 9.3.0（容器内 `ASCEND_HOME_PATH`） |
| TUNA 可达性 | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` 可达；pypi.org / github 不可达 |

## 2. 36 个算子依赖矩阵

| Operator | Layer | Backend | Python Dependencies | Source Dependencies | System Dependencies | TUNA Availability | AArch64 Wheel | Build Requirement | Environment | Recommended Source | Evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `allgather_fsdpturbo` | L3 | all_gather_experts_forward_fn | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.allgather_fsdpturbo::fsdp_turbo@2 |
| `allgather_hccl` | L1/L2/L3/L4 | torch.distributed.all_gather / reduce_scatter_te | `torch` | — | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64 | 否 | base | D | operators/allgather_hccl/backend.py（无可静态解析的第三方 import） |
| `allgather_megatron` | L3 | MoEAllGatherTokenDispatcher | `torch-npu` | `megatron`, `vendored-ascendep` | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：megatron | base | C+D | ep.allgather_megatron::megatron@2<br>ep.allgather_megatron::torch_npu@1 |
| `allgather_overlap_mindspeed` | L3 | MoELayerOverlapAllGather.apply | `torch` | `megatron`, `mindspeed`, `vendored-ascendep` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：megatron, mindspeed | base | C+D | ep.allgather_overlap_mindspeed::megatron@9<br>ep.allgather_overlap_mindspeed::mindspeed@7 |
| `alltoall_fsdpturbo` | L3 | all2all_experts_forward_fn (fused=False) | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.alltoall_fsdpturbo::fsdp_turbo@8 |
| `alltoall_hccl` | L1/L2/L3/L4 | torch.distributed.all_to_all_single | `torch-npu` | — | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64 | 否 | base | D | operators/alltoall_hccl/backend.py（无可静态解析的第三方 import） |
| `alltoall_megatron` | L3 | MoEAlltoAllTokenDispatcher | `torch-npu` | `megatron`, `vendored-ascendep` | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：megatron | base | C+D | ep.alltoall_megatron::megatron@2<br>ep.alltoall_megatron::torch_npu@1 |
| `alltoallv_gmm_cann` | L2/L3 | npu_alltoallv_gmm / npu_gmm_alltoallv | `torch`, `torch-npu` | `vendored-ascendep` | 共享 CANN + HCCL | torch:✓<br>torch-npu:✓ | torch:WHEEL_AARCH64<br>torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 否 | base | D+C | ep.alltoallv_gmm_cann::torch_npu@4 |
| `alltoallv_quant_gmm_cann` | L2 | npu_alltoallv_quant_gmm | `torch-npu` | — | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64 | 否 | base | D | torch_npu@11 |
| `chunkmoe_fsdpturbo` | L2/L3 | chunkmoe_experts_forward_fn | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.chunkmoe_fsdpturbo::fsdp_turbo@2 |
| `deepep_ascend` | L1/L2/L3 | deep_ep.EPBuffer | `torch` | `deep-ep` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | C+D | operators/deepep_ascend/backend.py（无可静态解析的第三方 import） |
| `deepep_cann` | L3 | cann_ops_transformer ElasticBuffer | `torch` | `cann-ops-transformer`, `vendored-ascendep` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：cann-ops-transformer | deep_ep_lueelu | C+D | ep.deepep_cann::cann_ops_transformer@36 |
| `deepep_direct_lueelu` | L1/L2/L3 | deep_ep.backends.direct.api.ElasticBuffer | — | `deep-ep` | 共享 CANN + HCCL | — | 源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | C | operators/deepep_direct_lueelu/backend.py（无可静态解析的第三方 import） |
| `deepep_lueelu` | L1/L2/L3 | deep_ep.ElasticBuffer (Mesh+Clos) | — | `deep-ep` | 共享 CANN + HCCL | — | 源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | C | operators/deepep_lueelu/backend.py（无可静态解析的第三方 import） |
| `deepep_sglang` | L3 | CamMoeDispatchNormal / CamMoeCombineNormal | `torch` | `deep-ep`, `vendored-ascendep` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | C+D | ep.deepep_sglang::deep_ep@50 |
| `deepep_v2_ascend` | L1/L2/L3 | deep_ep.ElasticBuffer | — | `deep-ep-v2` | 共享 CANN + HCCL | — | 源码依赖:需构建 | 是：deep-ep-v2 | deep_ep_v2 | C | deep_ep@11 |
| `dispatch_ffn_combine_sglang` | L3 | FuseMode.DISPATCH_FFN_COMBINE | `torch-npu` | `deep-ep`<br>⛔ 平台不支持（算子声明 operator:hardware） | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | F（operator:hardware 不支持 Ascend950） | operators/dispatch_ffn_combine_sglang/backend.py（无可静态解析的第三方 import） |
| `dispatch_ffn_combine_vllm_ascend` | L3 | torch.ops._C_ascend.dispatch_ffn_combine | `torch-npu` | `vllm-ascend`<br>⛔ 平台不支持（算子声明 operator:hardware） | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 否 | base | F（operator:hardware 不支持 Ascend950） | torch_npu@8<br>vllm_ascend@14 |
| `domino_fsdpturbo` | L3 | get_domino_experts_forward_fn | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.domino_fsdpturbo::fsdp_turbo@3 |
| `ep_gmm_fused_lueelu` | L2/L3 | deep_ep_experimental.ep_gmm_fused | `torch` | `deep-ep-experimental`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：deep-ep-experimental | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep-experimental | deep_ep_lueelu | C+D | ep.ep_gmm_fused_lueelu::deep_ep_experimental@6 |
| `fused_deep_moe_sglang` | L3 | FuseMode.FUSED_DEEP_MOE | `torch-npu` | `deep-ep` | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep | deep_ep_lueelu | C+D | operators/fused_deep_moe_sglang/backend.py（无可静态解析的第三方 import） |
| `fused_fsdpturbo` | L3 | all2all_experts_forward_fn (fused=True) | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.fused_fsdpturbo::fsdp_turbo@8 |
| `hierarchical_alltoallv_mindspeed` | L3/L4 | MoELayerOverlapAll2All with TP-extend-EP hierarc | `torch`, `torch-npu` | `megatron`, `mindspeed` | 共享 CANN + HCCL | torch:✓<br>torch-npu:✓ | torch:WHEEL_AARCH64<br>torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：megatron, mindspeed | base | C+D | torch_npu@21<br>megatron@22<br>mindspeed@24 |
| `mc2_combine_add_rmsnorm_cann` | L1 | npu_moe_distribute_combine_add_rms_norm | `torch-npu` | — | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64 | 否 | base | D | operators/mc2_combine_add_rmsnorm_cann/backend.py（无可静态解析的第三方 import） |
| `mc2_dispatch_cann` | L1/L2/L3 | npu_moe_distribute_dispatch_v2 / npu_moe_distrib | `torch`, `torch-npu` | `vendored-ascendep` | 共享 CANN + HCCL | torch:✓<br>torch-npu:✓ | torch:WHEEL_AARCH64<br>torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 否 | base | D+C | ep.mc2_dispatch_cann::torch_npu@4 |
| `mc2_dispatch_cann_v1` | L1/L2/L3 | npu_moe_distribute_dispatch / npu_moe_distribute | `torch-npu` | — | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64 | 否 | base | D | operators/mc2_dispatch_cann_v1/backend.py（无可静态解析的第三方 import） |
| `mc2_dispatch_cann_v3` | L1/L2/L3 | MoeDistributeBuffer / context-based aclnnMoeDist | — | `cann-ops-transformer` | 共享 CANN + HCCL | — | 源码依赖:需构建 | 是：cann-ops-transformer | base | C | cann_ops_transformer@20 |
| `mc2_fsdpturbo` | L3 | dispatcher_mc2 | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.mc2_fsdpturbo::fsdp_turbo@7 |
| `megamoe_cann` | L3 | cann_ops_transformer.ops.mega_moe (A16W16) | — | `cann-ops-transformer` | 共享 CANN + HCCL | — | 源码依赖:需构建 | 是：cann-ops-transformer | base | C | cann_ops_transformer@8 |
| `megamoe_deepgemm_ascend` | L3 | deep_gemm.fp8_fp4_mega_moe | `torch-npu` | `deepgemm-ascend` | 共享 CANN + HCCL | torch-npu:✓ | torch-npu:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deepgemm-ascend | base | C+D | deep_gemm@11<br>torch_npu@12 |
| `megamoe_fsdpturbo` | L3 | get_mega_moe_forward_fn | `torch` | `fsdp-turbo`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：fsdp-turbo | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：fsdp-turbo | base | C+D | ep.megamoe_fsdpturbo::fsdp_turbo@4 |
| `megamoe_lueelu` | L3 | bf16_mega_moe | `torch` | `deep-ep-experimental`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：deep-ep-experimental | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：deep-ep-experimental | deep_ep_lueelu | C+D | ep.megamoe_lueelu::deep_ep_experimental@3 |
| `megamoe_td` | L3 | MegaMoEFunction.apply | `torch` | `mega-moe`, `triton-distributed-ascend`, `vendored-ascendep`<br>⚠ 依赖文档未声明 Ascend950：mega-moe | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：mega-moe, triton-distributed-ascend | base | C+D+E | ep.megamoe_td::_moe_testkit@34<br>ep.megamoe_td::mega_moe@35 |
| `moonep` | L3 | npu_patch.Buffer.dispatch / combine (UDMA) | `torch` | `npu-patch`, `vendored-ascendep` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：npu-patch | moonep_npu | C+D | ep.moonep::npu_patch@10 |
| `moonep_dynamic_planner` | L1/L4 | MoonEP dev_npu planner / replica refresh | `torch` | `moonep-runtime`, `npu-patch`<br>⚠ 依赖文档未声明 Ascend950：moonep-runtime | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：moonep-runtime, npu-patch | moonep_npu | C+D | npu_patch@8<br>moonep@33 |
| `moonep_server_dedup` | L1/L4 | MoonEP dev_server_dedup dispatch/combine、权重预取和梯度 | `torch` | `moonep-dedup-runtime`, `npu-patch-dedup` | 共享 CANN + HCCL | torch:✓ | torch:WHEEL_AARCH64<br>源码依赖:需构建 | 是：moonep-dedup-runtime, npu-patch-dedup | moonep_dedup | C+D | npu_patch@10<br>moonep@45 |

## 3. TUNA 可获取性验证明细

用 A5 容器内 CPython 3.12.13 / aarch64 的真实标签核对 TUNA simple 索引（不下载大包，只解析 wheel 文件名）。

| manifest id | PyPI dist | 固定版本 | TUNA 有包 | 该版本 | Wheel 判定 | 样例文件 | 容器内已装 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `numpy` | `numpy` | 2.2.6 | 是 | 是 | WHEEL_AARCH64 | `numpy-2.2.6-cp312-cp312-manylinux_2_17_aarch64.manylinux2014_aarch64.whl` | 1.26.4 |
| `pyyaml` | `PyYAML` | 6.0.4 | 是 | 是 | WHEEL_AARCH64 | `pyyaml-6.0.4-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.manylinux_2_28_aarch64.whl` | 6.0.3 |
| `jsonschema` | `jsonschema` | 4.26.0 | 是 | 是 | WHEEL_PURE | `jsonschema-4.26.0-py3-none-any.whl` | 4.26.0 |
| `pandas` | `pandas` | 2.2.3 | 是 | 是 | WHEEL_AARCH64 | `pandas-2.2.3-cp312-cp312-manylinux2014_aarch64.manylinux_2_17_aarch64.whl` | 3.0.6 |
| `matplotlib` | `matplotlib` | 3.10.9 | 是 | 是 | WHEEL_AARCH64 | `matplotlib-3.10.9-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl` | 未安装 |
| `torch` | `torch` | 2.10.0 | 是 | 是 | WHEEL_AARCH64 | `torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl` | 2.10.0+cpu |
| `torch-npu` | `torch-npu` | 2.10.0.post4 | 是 | 是 | WHEEL_AARCH64 | `torch_npu-2.10.0.post4-cp312-cp312-manylinux_2_28_aarch64.whl` | 2.10.0.post7 |
| `torch-npu(latest-in-container)` | `torch-npu` | 2.10.0.post7 | 是 | 是 | WHEEL_AARCH64 | `torch_npu-2.10.0.post7-cp312-cp312-manylinux_2_28_aarch64.whl` | 未安装 |
| `shmem` | `shmem` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `megatron` | `megatron` | 未固定 | 是 | — | WHEEL_PURE | `Megatron-0.1.0-py3-none-any.whl` | 未安装 |
| `megatron-core` | `megatron-core` | 未固定 | 是 | — | WHEEL_AARCH64 | `megatron_core-0.14.0-cp312-cp312-manylinux_2_24_aarch64.manylinux_2_28_aarch64.whl` | 未安装 |
| `mindspeed` | `mindspeed` | 未固定 | 是 | — | SDIST_ONLY | `mindspeed-0.0.1.dev1.tar.gz` | 未安装 |
| `deep-ep` | `deep-ep` | 未固定 | 是 | — | SDIST_ONLY | `deep_ep-1.0.0.tar.gz` | 未安装 |
| `deep-ep-v2` | `deep-ep-v2` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `deep-ep-experimental` | `deep-ep-experimental` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `fsdp-turbo` | `fsdp-turbo` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `npu-patch` | `npu-patch` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `deepgemm-ascend` | `deep-gemm` | 未固定 | 是 | — | SDIST_ONLY | `deep_gemm-1.0.0.tar.gz` | 未安装 |
| `deepgemm-ascend(alt)` | `deepgemm` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `moonep` | `moonep` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `cann-ops-transformer` | `cann-ops-transformer` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 1.0.0 |
| `vllm-ascend` | `vllm-ascend` | 未固定 | 是 | — | WHEEL_AARCH64 | `vllm_ascend-0.21.0rc1-cp312-cp312-manylinux_2_34_aarch64.whl` | 0.19.1rc2.dev4004+g07fb73958 |
| `triton-dist` | `triton-distributed` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `triton-dist(alt)` | `triton-distributed-ascend` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `mega-moe` | `mega-moe` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |
| `mega-moe(alt)` | `mega_moe` | 未固定 | 否 | — | NOT_ON_TUNA | `—` | 未安装 |

判定含义：`WHEEL_AARCH64` 有 cp312/abi3 aarch64 wheel；`WHEEL_PURE` 纯 Python wheel；`SDIST_ONLY` 只有源码包（需 A5 构建）；`NOT_ON_TUNA` TUNA 无此包。

## 4. 第三方源码清单

| manifest id | 上游 | 固定 commit/tag | 在 TUNA？ | 可用 PyPI 同名包替代？ | 需 A5 构建 |
| --- | --- | --- | --- | --- | --- |
| `megatron` | https://github.com/NVIDIA/Megatron-LM | baf94af | 否 | 否（`megatron-core` 是不同发行版） | 是（editable） |
| `mindspeed` | https://gitcode.com/Ascend/MindSpeed | c68e201 | 有同名 stub 0.0.1.dev1 | 否（stub 非 Ascend MindSpeed） | 是（editable） |
| `deep-ep` | https://github.com/lueelu/DeepEP | 83252a5 | 有同名 1.0.0 sdist | 否（非 lueelu fork） | 是（build.sh + pip） |
| `deep-ep-v2` | https://gitcode.com/lishaoxun/DeepEP_9283 | 69090a1 | 否 | 否 | 是（build.sh + pip） |
| `deep-ep-experimental` | 同上 fork 的 experiments 子目录 | 83252a5 | 否 | 否 | 是（子目录 pip） |
| `fsdp-turbo` | https://gitcode.com/Ascend/FSDPTurbo | 474a39a | 否 | 否 | 是（editable；950 需实测） |
| `npu-patch / npu-patch-dedup` | https://gitcode.com/Ascend/MoonEP | b4127ba / d8495c4 | 否 | 否 | 是（pip install ./npu_patch） |
| `moonep-runtime / moonep-dedup-runtime` | https://gitcode.com/Ascend/MoonEP | b4127ba / d8495c4 | 否 | 否 | 是（editable） |
| `deepgemm-ascend` | https://github.com/deepseek-ai/DeepGEMM-Ascend | 3a35326 | 有同名 deep-gemm 1.0.0 sdist | 否（非 DeepGEMM-Ascend） | 是（pip install .） |
| `cann-ops-transformer` | https://gitcode.com/cann/ops-transformer | 68a67e1 | 否 | 否 | 是（pip install --no-build-isolation .） |
| `vllm-ascend` | https://github.com/vllm-project/vllm-ascend | 0f63cfa | 有 0.21.0rc1 aarch64 wheel | 否（版本与 pin 不同） | 否（容器已带 dev 构建） |
| `mega-moe` | https://gitcode.com/jzhoujg/Mega-MoE-TD | 41d4793 | 否 | 否 | 是（editable；950 需实测） |
| `shmem` | https://gitcode.com/cann/shmem | tag 8.2.RC1 | 否 | 否 | 是（build.sh + wheel） |
| `triton-distributed-ascend` | https://gitcode.com/Ascend/Triton-distributed-ascend | 未固定 | 否 | 否 | 未确定 |

已 vendored，无需下载：`vendored-ascendep`（`operators/vendor/ascendep`，23 文件，commit 24ff84d，`python scripts/vendor_ascendep.py --check` 在 A5 上 exit 0）。

## 5. Native Build 清单

| 项 | 源码位置 | 构建命令 | 工具链 | 必须在 A5 构建 |
| --- | --- | --- | --- | --- |
| `asc-comm-kernels` | `kernels/asc_comm` | `bash kernels/asc_comm/build.sh --soc Ascend950` | cmake + g++ + CANN ASC-COMM | 是（A5 已构建成功，manifest 记录 AIV/CCU `built:false`，设备源缺失） |
| `l0-native-bridges` | `baselines/l0/transport/native` | `bash baselines/l0/transport/native/build.sh` | g++ | 是（未构建，LINK-001 因此 unsupported） |
| `cann-ops-transformer` | gitcode 源码 | `pip install --no-build-isolation .` | CANN + ccec | 是 |
| `shmem` | gitcode 源码 | `bash build.sh` + wheel | CANN | 是 |
| `deep-ep` / `deep-ep-v2` | fork 源码 | `bash build.sh` + `pip install .` | CANN + C++ | 是 |
| `npu_patch` | MoonEP 源码 | `pip install --no-build-isolation ./npu_patch` | CANN | 是 |
| `deepgemm-ascend` | DeepGEMM-Ascend | `pip install --no-build-isolation .` | CANN | 是 |

## 6. CANN / 系统提供（不得由 benchmark 替换）

| 项 | 提供方 | 说明 |
| --- | --- | --- |
| `npu-driver` | container | >=24.1.RC1 |
| `cann-toolkit` | container | >=9.0.0,<9.3 |
| `hccl` | container | >=8.1.RC1 |
| `hcomm` | container | >=8.1.RC1 |
| `torch` | container | ==2.10.0 |
| `torch-npu` | container | ==2.10.0.post4 |
| `npu-smi` | container | >=24.1.RC1 |
| `ascendc-toolchain` | container | >=8.1.RC1 |
| `msprof` | host | toolkit-managed |
| `hccl-test` | host | toolkit-managed |
| `hccl-config` | host | toolkit-managed |
| `multinode-env` | host | deployment-managed |
| `device-permissions` | container | driver-managed |
| `asc-comm-build-toolchain` | host | >=3.20 (cmake) + any C++17 g++ + the CAN |

HCCL 与 HCOMM 均由 CANN 提供：本轮未把它们计入第三方源码，也未建议任何自动替换运行时的动作。

## 7. 现有 manifest 的问题

1. **声明但未被任何算子使用**：`allgather_hccl` 的 `required_by` 含 `vendored-ascendep`，但它走 `CollectiveBackend("allgather")`，不加载 ascendep。同类可疑项见第 8 节表格的 “—” 行。
2. **实际 import 但未在 manifest 建立 operator 关联**：`dispatch_ffn_combine_sglang` / `fused_deep_moe_sglang` 的第三方 import 在 `operators/sglang_fused.py` 里动态发生，`backend.py` 静态扫描为空，manifest 靠 `required_by` 手工登记（deep-ep + torch-npu）。
3. **同名不同源风险**：`mindspeed`（TUNA 0.0.1.dev1）、`deep-ep`（TUNA 1.0.0）、`deep-gemm`（TUNA 1.0.0）、`megatron-core`（TUNA 0.14.0）都与 pin 的上游 fork 不是同一项目，不可替代。
4. **版本漂移**：容器实测 numpy 1.26.4≠2.2.6、pandas 3.0.6≠2.2.3、pyyaml 6.0.3≠6.0.4、torch-npu 2.10.0.post7≠post4，doctor 会对这些报 warn。
5. **隔离环境继承**：`deep_ep_lueelu` / `deep_ep_v2` / `moonep_npu` / `moonep_dedup` 四个独立环境需要继承 torch/torch_npu；容器当前只装了一套 torch 2.10.0+cpu，独立 venv 尚未建立。
6. **HCCL 未被误列为第三方**：manifest 中 `hccl` 的 `kind` 是 `host_provided`，正确。
7. **HCOMM 无普通安装流程**：`hcomm` 同为 `host_provided`，未发现可替换运行时的安装步骤，正确。
8. **DeepGEMM-Ascend 缺独立依赖定义**：manifest 只有 `deepgemm-ascend` 一条，没有区分「PyPI 同名 deep-gemm」与「Ascend fork」，容易误导安装脚本。
9. **两个 DeepEP fork 模块名冲突**：`deep-ep` 与 `deep-ep-v2` 都装成 `deep_ep`，`deep-ep-experimental` 又复用 lueelu fork；三者只能靠独立 venv 隔离。
10. **`npu_patch` 两条记录同模块名**：`npu-patch` 与 `npu-patch-dedup` 都提供 `npu_patch`，分属 `moonep_npu` 与 `moonep_dedup` 两个环境，不能在同一个 venv 共存。
11. **CANN 版本超出声明区间**：manifest 声明 `cann-toolkit >=9.0.0,<9.3`，A5 容器实际是 **9.3.0**（`ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.3.0`）。doctor 对 CANN 的判定需要按实际版本核对，否则会把已就绪的栈报成越界。

## 8. 推荐依赖重构方案（逐依赖）

策略代号：A TUNA 直接装；B TUNA sdist 本地构建；C third_party 源码 + A5 构建；D CANN/容器提供；E 离线 bundle；F 平台不支持不装。

| manifest id | 策略 | 说明 |
| --- | --- | --- |
| `asc-comm-kernels` | C | in-repo kernels/asc_comm; needs CANN ASC-COMM toolchain |
| `cann-ops-transformer` | C | gitcode.com/cann/ops-transformer @68a67e1; container already has 1.0.0 |
| `deep-ep` | C | github lueelu/DeepEP @83252a5; TUNA 'deep-ep 1.0.0' is unrelated |
| `deep-ep-experimental` | C | lueelu fork experiments subdirs; release notes omit 950, confirm with a run |
| `deep-ep-v2` | C | gitcode lishaoxun/DeepEP_9283 @69090a1; not on TUNA |
| `deepgemm-ascend` | C | github DeepGEMM-Ascend @3a35326; TUNA 'deep-gemm 1.0.0' is unrelated |
| `fsdp-turbo` | C | gitcode FSDPTurbo @474a39a; release notes omit 950, confirm with a real run |
| `jsonschema` | A | TUNA wheel; container already at pin |
| `l0-native-bridges` | C | in-repo baselines/l0/transport/native/build.sh |
| `matplotlib` | A | TUNA wheel; optional, not installed |
| `mega-moe` | C | gitcode Mega-MoE-TD @41d4793; release notes omit 950, confirm with a real run |
| `megatron` | C | NVIDIA Megatron-LM @baf94af; PyPI 'megatron-core' is a different distribution |
| `mindspeed` | C | gitcode MindSpeed @c68e201; TUNA 'mindspeed 0.0.1.dev1' is an unrelated stub |
| `moonep-dedup-runtime` | C | gitcode MoonEP dev_server_dedup @d8495c4; not on TUNA |
| `moonep-runtime` | C | gitcode MoonEP @b4127ba; not on TUNA |
| `mpi` | E | system package manager needs sudo; offline bundle when there is no network |
| `npu-patch` | C | gitcode MoonEP @b4127ba; not on TUNA |
| `npu-patch-dedup` | C | gitcode MoonEP dev_server_dedup @d8495c4; not on TUNA |
| `numpy` | A | TUNA wheel, but container has 1.26.4 vs pin 2.2.6 |
| `pandas` | A | TUNA wheel; optional dep, pins differ (3.0.6 vs 2.2.3) |
| `pyyaml` | A | TUNA wheel; container 6.0.3 vs pin 6.0.4 |
| `shmem` | C | not on TUNA; gitcode.com/cann/shmem tag 8.2.RC1, needs build.sh |
| `torch` | D | container/CANN stack owns it; TUNA wheel exists but must not replace it |
| `torch-npu` | D | container owns it; pin post4 vs container post7 |
| `triton-distributed-ascend` | E | no dist metadata in container, not on TUNA, unpinned |
| `vendored-ascendep` | C | already vendored in operators/vendor/ascendep (24ff84d) |
| `vllm-ascend` | D | container ships a dev build; TUNA has 0.21.0rc1 (different pin) |

## 9. 未验证事项与阻塞

- **只验证了 TUNA，没有访问 pypi.org 或 GitHub**：所有“不在 TUNA”的条目仅代表 TUNA 侧缺失，不代表上游不存在。
- **未实际安装任何包**：`WHEEL_AARCH64` 只表示 TUNA 存在匹配标签的文件名，未做下载安装验证。
- **独立 venv 未建立**：4 个隔离环境（deep_ep_lueelu / deep_ep_v2 / moonep_npu / moonep_dedup）在 A5 上还不存在，无法验证「在对应 venv 里 doctor 能否检查实际依赖」。
- **`triton-distributed-ascend` 未固定版本**，容器内可 import 但无 dist 元数据，来源不明，标记 UNKNOWN。
- **ASC-COMM 设备源缺失**：A5 构建成功但只编译了 host 部分，manifest 已如实记录 AIV/CCU `built:false`（原因：`aiv/write_nbi/kernel.cpp`、`ccu/dsl/kernel.cpp` 不存在）。

## 10. 最终结论

以 TUNA 为默认 Python 来源、third_party 为特殊源码来源、A5 为 Native Build 环境：

| 类别 | 算子数 | 说明 |
| --- | ---: | --- |
| 依赖可闭合（TUNA wheel 即可） | 5 | 主要是 HCCL 集合通信与 cann_special 类适配器 |
| 依赖可闭合（需 third_party + A5 构建） | 29 | 需把源码纳入 third_party 或预置到 source-cache |
| 平台阻塞（算子级 hardware 约束排除 Ascend950） | 2 | 装也无法运行，应标记 BLOCKED_PLATFORM |
| 依赖文档未列出 Ascend950（需实测确认） | 11 | 依赖可装可建，装完后用真机跑一次才能定性 |
| 仅系统/CANN 提供 | 0 | — |
| **合计** | **36** | 与 registry 的 36 个算子一致（后两行为附加标记，不重复计数） |

**平台阻塞**（算子自身 `constraints.hardware` 排除 Ascend950，装也无效）：`dispatch_ffn_combine_sglang`, `dispatch_ffn_combine_vllm_ascend`。

**依赖文档未声明 Ascend950（不是平台阻塞，需装完后实测）**：`allgather_fsdpturbo`, `alltoall_fsdpturbo`, `chunkmoe_fsdpturbo`, `domino_fsdpturbo`, `ep_gmm_fused_lueelu`, `fused_fsdpturbo`, `mc2_fsdpturbo`, `megamoe_fsdpturbo`, `megamoe_lueelu`, `megamoe_td`, `moonep_dynamic_planner`。

**依赖可获取 ≠ 算子能运行**：以本次 A5 实测为例，`megamoe_cann` 的依赖（`cann_ops_transformer`）已就绪，运行仍因 `dispatch_quant_out_dtype` 不支持而失败；`mc2_dispatch_cann_v1` 依赖齐备但返回元组 arity 不符。反之 `deepep_cann` 依赖齐备且真机通过。依赖盘点只能回答前者。

现存最大缺口不是平台能力，而是**获取渠道**：29 个算子需要 GitHub/gitcode 源码，而 A5 无法访问外网（只能预置 source-cache 走 `--offline`）；另有 11 个算子的依赖发布说明尚未覆盖 950DT，装完后必须用真机跑一次才能定性；真正的平台阻塞只有 2 个（`dispatch_ffn_combine_sglang`、`dispatch_ffn_combine_vllm_ascend` 仅声明 Atlas A3）。

