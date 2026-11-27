#!/usr/bin/env bash
set +e
ENV=/home/vllm/x00886548/batch_inv_glm52/envs/vllm312
PY=$ENV/bin/python
rm -rf /root/ep-bench/04_scripts
mkdir -p /root/ep-bench
tar -xzf /root/ep04.tgz -C /root/ep-bench
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="$ENV/bin:$PATH"
export PYTHONPATH=/root/ep-bench/04_scripts
export OMP_NUM_THREADS=1
cd /root/ep-bench/04_scripts || exit 1
RUN=20261127T011813-36cd31e2
OUT=/root/ep-bench/results/npu-smoke1
"$PY" run_benchmark.py analyze --run-id "$RUN" --output "$OUT" > /tmp/analyze.out 2>&1
echo "analyze exit: $?"
"$PY" - <<PYCODE
import json
data = json.load(open("/root/ep-bench/results/npu-smoke1/20261127T011813-36cd31e2/analysis/results.json"))
metrics = data.get("coverage_metrics") or {}
print({key: metrics.get(key) for key in
       ("planned", "executed", "correct", "verified",
        "execution_coverage", "correctness_pass_rate", "verified_performance_coverage")})
PYCODE
"$PY" run_benchmark.py report --run-id "$RUN" --output "$OUT" > /tmp/report.out 2>&1
echo "report exit: $?"
REPORT=$(find "$OUT/$RUN" -name "benchmark_report.md" | head -1)
grep -n "2.8 Coverage metrics" -A 8 "$REPORT" | head -11
