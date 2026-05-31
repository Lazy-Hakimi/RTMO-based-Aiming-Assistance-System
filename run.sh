#!/bin/bash
# RTMO AimBot 启动脚本

set -e

echo "=================================="
echo "RTMO AimBot Launcher"
echo "=================================="

# 检查 root 权限
if [ "$EUID" -ne 0 ]; then
    echo "警告: 未以 root 运行，HID 设备可能需要额外权限"
    echo "建议: sudo ./run.sh"
fi

# 加载 uinput 模块
if ! lsmod | grep -q uinput; then
    echo "加载 uinput 模块..."
    modprobe uinput
fi

# 设置权限
chmod 666 /dev/uinput 2>/dev/null || true
chmod 666 /dev/video0 2>/dev/null || true

# 检查引擎文件
if [ ! -f "models/rtmo_l_640x640.trt" ]; then
    echo "错误: 找不到 TensorRT 引擎文件"
    echo "请先运行: python3 scripts/onnx2trt.py --onnx models/rtmo_l_640x640.onnx --output models/rtmo_l_640x640.trt"
    exit 1
fi

# 运行主程序
echo "启动 AimBot..."
python3 main.py "$@"
