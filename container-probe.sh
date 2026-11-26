echo "== os"
. /etc/os-release 2>/dev/null; echo "$PRETTY_NAME"
echo "== pythons"
ls /usr/bin/python3.* 2>/dev/null
for p in python3.10 python3.11 python3.12 python3.13; do
  if command -v "$p" >/dev/null 2>&1; then "$p" -V; fi
done
echo "== ascend"
ls /usr/local/Ascend
echo "== version"
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null | head -5
cat /usr/local/Ascend/ascend-toolkit/version.cfg 2>/dev/null | head -5
echo "== toolkit dirs"
ls /usr/local/Ascend/ascend-toolkit 2>/dev/null | head
ls /usr/local/Ascend/cann 2>/dev/null | head
echo "== env scripts"
find /usr/local/Ascend -maxdepth 4 -name set_env.sh 2>/dev/null | head -5
echo "== torch"
python3 - <<'PY'
try:
    import torch
    print("torch", torch.__version__)
except Exception as exc:
    print("torch import failed:", exc)
try:
    import torch_npu
    print("torch_npu", torch_npu.__version__)
except Exception as exc:
    print("torch_npu import failed:", exc)
PY
echo "== npu"
npu-smi info 2>/dev/null | head -8
echo "== soc"
npu-smi info -t board -i 0 2>/dev/null | grep -Ei "board|chip|soc|name" | head -5
echo "== devices"
ls /dev/davinci* 2>/dev/null | head -3
ls /dev/davinci* 2>/dev/null | wc -l
echo "== hccl"
ldconfig -p 2>/dev/null | grep -E "libhccl|libhcomm" | head -4
