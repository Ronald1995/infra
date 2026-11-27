"""Write or verify the generated capability documentation.

    python scripts/docgen.py --check          # fail when a README drifted
    python scripts/docgen.py --write          # regenerate every README + the matrix

The data comes from `common.docgen`, which asks the scheduler.  Nothing is inferred
from a case ID's prefix, from a level name, or from a hand-written table.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common import docgen                                        # noqa: E402
from common.registry import registry                             # noqa: E402

OPERATORS = ROOT / "operators"
MATRIX = OPERATORS / "capability_matrix.md"


def readme_path(operator):
    return OPERATORS / operator / "README.md"


def matrix_text(plans):
    lines = ["# EP 算子能力矩阵（自动生成）", "",
             "此文件由 `python scripts/docgen.py --write` 生成，请勿手工编辑。",
             "数据来源：`operators/registry.yaml` 的能力声明 + `common/case_spec.py` 的",
             "Case 定义 + `common/scheduler.py` 的实际判定。", "",
             "## 层次总表", ""]
    lines += docgen.render_matrix(plans)
    lines += ["", "## Case 支持矩阵", ""]
    lines += docgen.render_case_matrix(plans)
    lines += ["", "## 不支持原因", ""]
    lines += docgen.render_unsupported(plans)
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="regenerate the documents")
    action.add_argument("--check", action="store_true", help="fail when they drifted")
    parser.add_argument("--only", type=lambda v: [i.strip() for i in v.split(",") if i.strip()])
    parser.add_argument("--matrix-only", action="store_true")
    args = parser.parse_args(argv)

    specs = registry()
    operators = args.only or sorted(specs)
    unknown = [name for name in operators if name not in specs]
    if unknown:
        raise SystemExit(f"unknown operator(s): {', '.join(unknown)}")

    plans = {name: docgen.plan_for(name) for name in operators}
    matrix = matrix_text(plans)

    changed, stale = [], []
    if not args.matrix_only:
        for name in operators:
            path = readme_path(name)
            if not path.is_file():
                stale.append(f"{name}: README missing")
                continue
            original = path.read_text(encoding="utf-8")
            updated = docgen.splice_contract(original, docgen.contract_section(
                name, specs[name], plans[name]))
            if updated != original:
                changed.append(name)
                if args.write:
                    path.write_text(updated, encoding="utf-8")

    current = MATRIX.read_text(encoding="utf-8") if MATRIX.is_file() else ""
    matrix_changed = current != matrix
    if args.write and matrix_changed:
        MATRIX.write_text(matrix, encoding="utf-8")

    if args.write:
        print(f"updated {len(changed)} README(s): {', '.join(changed) or 'none'}")
        print(f"capability matrix: {'rewritten' if matrix_changed else 'unchanged'}")
        return 0

    problems = list(stale)
    if changed:
        problems.append(f"{len(changed)} README(s) out of date: {', '.join(changed)}")
    if matrix_changed:
        problems.append("operators/capability_matrix.md is out of date")
    if problems:
        for problem in problems:
            print(f"STALE: {problem}", file=sys.stderr)
        print("run: python scripts/docgen.py --write", file=sys.stderr)
        return 1
    print("all generated capability documentation is up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
