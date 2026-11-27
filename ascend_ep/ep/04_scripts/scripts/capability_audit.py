"""Write the operator capability audit: what each adapter can really support.

    python scripts/capability_audit.py [--out reports/operator_capability_audit.md]

Everything is derived from three declarations -- the registry capabilities, the case
catalog, and the scheduler's verdict -- so the audit cannot drift from the code.
The one thing it does *not* invent is the L0 primitive mapping: that field is
printed as recorded, including `unknown`, because guessing a transport path from an
operator's name is exactly what this refactor must avoid.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import docgen                                          # noqa: E402
from common.case_spec import catalog                              # noqa: E402
from common.registry import registry                              # noqa: E402

DEFAULT_OUT = ROOT.parent.parent / "reports" / "operator_capability_audit.md"


def _l0_mapping(spec):
    """The registry's declared underlying path, or `unknown` with the reason."""
    path = getattr(spec, "underlying", None) or {}
    if not path:
        return "unknown（registry 未声明底层原语，需按源码补齐）"
    parts = []
    for key in ("primitives", "transport", "resources"):
        values = path.get(key) or ()
        parts.append(f"{key}={', '.join(values) or '—'}")
    return "；".join(parts)


def build(plans=None):
    plans = plans or docgen.plan_all()
    specs = registry()
    cases = catalog()
    lines = ["# EP 算子能力审计（自动生成）", "",
             "由 `python scripts/capability_audit.py` 生成。数据来自 "
             "`operators/registry.yaml` 的能力声明、`common/case_spec.py` 的 Case "
             "定义，以及 `common/scheduler.py` 在 `configs/ep8_bf16.yaml` 下的实际判定。",
             "",
             "生成说明：`conditional` 表示参考配置未选中该场景、但换成匹配配置即可运行；"
             "`unsupported` 表示能力或约束不满足；`not_implemented` 表示已识别但未实现。", ""]

    lines += ["## 1. 能力声明与层次覆盖", "",
              "| Operator | 声明能力 | L1 | L2 Serial | L2 Native | L3 | L4 | L0 底层映射 |",
              "|---|---|---|---|---|---|---|---|"]
    for operator in sorted(plans):
        spec = specs[operator]
        row = docgen.summary_row(operator, plans[operator])
        cells = []
        for key in ("L1", "L2S", "L2N", "L3", "L4"):
            text = docgen._levels(row[key])
            if row[key + "_conditional"]:
                extra = docgen._levels(row[key + "_conditional"])
                text = f"{text}（条件 {extra}）" if text != "—" else f"（条件 {extra}）"
            cells.append(text)
        lines.append(f"| `{operator}` | {', '.join(spec.capabilities) or '—'} | "
                     + " | ".join(cells) + f" | {_l0_mapping(spec)} |")

    lines += ["", "## 2. 逐 Case 判定", ""]
    lines += docgen.render_case_matrix(plans)

    lines += ["", "## 3. 不可运行原因汇总", ""]
    lines += docgen.render_unsupported(plans)

    lines += ["", "## 4. 层次定义对照", "",
              "| 层次 | 测试对象 | 入口 |", "|---|---|---|",
              "| L0 | 硬件与基础能力（HBM/P2P/HCCL/SHMEM/GMM/SwiGLU/并发），与具体 EP 算子解耦 | "
              "`benchmarks/l0_hardware.py` → `baselines/l0/` |",
              "| L1 | 独立通信接口（Dispatch/Combine/缓存/异步/量化/去重/规划） | "
              "`benchmarks/cases/l1_communication.py` |",
              "| L2 | 通信与计算融合：串行组合（FUSE-001/003）与原生融合（其余 FUSE） | "
              "`benchmarks/cases/l2_fusion.py` |",
              "| L3 | 完整 MoE 执行（前向/反向/重算/并发/多层/动态 token/trace replay） | "
              "`benchmarks/cases/l3_moe.py` |",
              "| L4 | 路由器、动态规划与去重层次 | `benchmarks/cases/l4_router_planner.py` |", ""]

    lines += ["## 5. 本次审计发现的结构性问题", ""]
    findings = []
    for case_id, case in sorted(cases.items()):
        if case.alias_of is not None:
            continue
        if not case.requires and not case.conditional_requires:
            findings.append(f"- `{case_id}` 既无 `requires` 也无条件能力，静态分析会误判为"
                            f"全算子可运行。")
    findings.append("- L0 与算子的底层映射尚未按源码逐项核实，`registry.yaml` 中的 "
                    "`underlying` 字段保持 `unknown`，不按算子名称推断传输路径。")
    findings.append("- COMM-011 的计时区间不包含准备阶段 Dispatch，已改为 `setup_requires` 表达。")
    findings.append("- MOE-002/MOE-004 的前向图在 setup 构建，已改为 `setup_requires` 表达。")
    lines += findings + [""]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    text = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
