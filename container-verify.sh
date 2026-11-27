#!/usr/bin/env bash
set +e
PY=/home/vllm/x00886548/batch_inv_glm52/envs/vllm312/bin/python
rm -rf /root/ep-bench/04_scripts
mkdir -p /root/ep-bench
tar -xzf /root/ep04.tgz -C /root/ep-bench
cd /root/ep-bench/04_scripts || exit 1
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PYTHONPATH=/root/ep-bench/04_scripts
"$PY" scripts/doctor.py --profile npu > /tmp/doctor.out 2>&1
head -30 /tmp/doctor.out
echo "--- summary"
tail -2 /tmp/doctor.out
