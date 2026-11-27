set +e
echo "== before"; readlink /usr/local/Ascend/ascend-toolkit/latest
if [ ! -e /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  echo "broken: repointing latest -> /usr/local/Ascend/cann-9.0.0"
  ln -sfn /usr/local/Ascend/cann-9.0.0 /usr/local/Ascend/ascend-toolkit/latest
fi
echo "== after"; readlink -f /usr/local/Ascend/ascend-toolkit/latest; ls -l /usr/local/Ascend/ascend-toolkit/set_env.sh
echo "== env after source"
bash -c 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; echo "ASCEND_HOME_PATH=$ASCEND_HOME_PATH"; echo "ASCEND_TOOLKIT_HOME=$ASCEND_TOOLKIT_HOME"; echo "LD=$LD_LIBRARY_PATH" | cut -c1-160; command -v ccec; ldconfig -p 2>/dev/null | grep -c libhccl'
echo "== version.cfg"
ls /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null && head -3 /usr/local/Ascend/ascend-toolkit/latest/version.cfg
echo "== torch_npu import with the toolkit env"
PY=/home/vllm/x00886548/batch_inv_glm52/envs/vllm312/bin/python
bash -c "source /usr/local/Ascend/ascend-toolkit/set_env.sh; '$PY' - <<'PYCODE'
import torch
print('torch', torch.__version__)
import torch_npu
print('torch_npu', torch_npu.__version__)
print('npu available:', torch.npu.is_available(), 'count:', torch.npu.device_count())
if torch.npu.is_available():
    print('device 0:', torch.npu.get_device_name(0))
PYCODE"
echo "== exit: $?"
