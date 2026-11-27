set +e
echo "== libhccl / libhcomm under Ascend"; find /usr/local/Ascend -name "libhccl.so*" -o -name "libhcomm.so*" 2>/dev/null | head -8
echo "== version files under cann-9.0.0"; find /usr/local/Ascend/cann-9.0.0 -maxdepth 2 -name "version*" 2>/dev/null | head -8
echo "== ascend version.info"; cat /usr/local/Ascend/version.info 2>/dev/null
echo "== toolkit install info"; find /usr/local/Ascend -maxdepth 3 -name "ascend_toolkit_install.info" 2>/dev/null | head -3
echo "== latest"; readlink -f /usr/local/Ascend/ascend-toolkit/latest
echo "== lib64 sample"; ls /usr/local/Ascend/cann-9.0.0/lib64/ 2>/dev/null | head -8
echo "== hccl in driver?"; ls /usr/local/Ascend/driver/lib64/ 2>/dev/null | grep -i hccl | head -3
