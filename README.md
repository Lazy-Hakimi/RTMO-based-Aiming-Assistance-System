# RTMO AimBot Enhanced - 增强版人体追踪辅助瞄准系统

基于 **RTMO-l** (Real-Time Multi-person pose estimation Optimized) 姿态估计模型的增强版AimBot系统，部署于 **Jetson AGX Xavier 32GB**，通过 **ESP32-S3** 模拟HID鼠标设备，实现低延迟、高精度的多目标追踪瞄准。

---

## 一、系统架构

```
                    ┌─────────────────┐
                    │   游戏主机       │
                    │ (HDMI输出)      │
                    └────────┬────────┘
                             │ HDMI
                             ▼
                    ┌─────────────────┐
                    │  低延迟采集卡     │
                    │  (V4L2 1080p60) │
                    └────────┬────────┘
                             │ USB
                             ▼
┌──────────────────────────────────────────────────────┐
│            Jetson AGX Xavier 32GB (推理端)            │
│  ┌────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │CaptureThread│  │InferThread│  │   MainThread      │ │
│  │ V4L2采集   │  │PyTorch/    │  │ 可视化/校准UI     │ │
│  │            │  │TensorRT推理│  │ 键盘交互控制      │ │
│  │            │  │RTMO解码   │  │                   │ │
│  │            │  │身体朝向估算│  │                   │ │
│  │            │  │多目标优先级│  │                   │ │
│  │            │  │压枪补偿   │  │                   │ │
│  │            │  │准星校准   │  │                   │ │
│  └─────┬──────┘  └────┬─────┘  └──────────────────┘ │
│        │              │                                │
│        └──────┬───────┘                                │
│               ▼                                        │
│  ┌─────────────────────────────────────────────────┐  │
│  │              ESP32Bridge (UART/USB-CDC)          │  │
│  │         二进制协议编码/解码 + 心跳检测             │  │
│  └────────────────────────┬────────────────────────┘  │
└───────────────────────────┼────────────────────────────┘
                            │ UART / USB-CDC
                            ▼
┌──────────────────────────────────────────────────────┐
│              ESP32-S3 DevKitC (HID设备端)              │
│  ┌──────────┐  ┌────────────┐  ┌──────────────────┐  │
│  │  UART    │  │ 协议解码    │  │  USBHIDMouse     │  │
│  │ 接收器    │──▶│ 状态机      │──▶│  鼠标事件输出     │  │
│  └──────────┘  └────────────┘  └──────────────────┘  │
│                                                         │
│  TinyUSB协议栈 / USB-OTG模式 / 1000Hz报告率             │
└──────────────────────────────────────────────────────┘
                            │
                            │ USB
                            ▼
                    ┌─────────────────┐
                    │    游戏主机      │
                    │ (USB鼠标输入)   │
                    └─────────────────┘
```

---

## 二、新增功能清单

### 2.1 双后端推理引擎 (新增)
支持两种推理后端，可根据场景灵活选择:

| 后端 | 默认 | 推理速度 | 精度 | 适用场景 |
|------|------|----------|------|----------|
| **PyTorch (.pth)** | 是 | ~25-35ms | 最高 | 开发调试、快速部署 |
| **TensorRT (.trt)** | 否 | ~15-25ms | 高 | 生产环境、极致性能 |

**PyTorch 原生推理优势:**
- 直接使用 MMPose 官方发布的 `.pth` 权重文件，无需转换
- 支持 FP16 半精度推理 (AMP自动混合精度)
- 代码简洁，便于调试和二次开发
- 自动兼容 MMPose 模型更新

**TensorRT 推理优势:**
- 极致推理性能，延迟更低
- 适合生产环境长期稳定运行
- 内存占用更小

### 2.2 身体/枪口朝向估算
- **肩线向量法**: 通过左右肩关键点构建参考向量，分析肩线与水平面的夹角
- **髋线验证法**: 使用左右髋关键点辅助验证朝向判断
- **宽高比分析法**: 肩宽/髋宽比例区分正面/侧面/背面
- **8方向分类**: 正面、背面、左侧面、右侧面 + 4个斜向
- **时间域平滑**: 3帧历史数据平滑，减少抖动

### 2.3 多目标优先级选择策略
基于综合权重公式选择最优目标：
```
Score = w_dist * score_dist + w_threat * score_threat + 
        w_ori * score_ori + w_size * score_size + w_conf * score_conf
```
- **距离分数** (35%): 离准星越近分越高，高斯衰减
- **威胁度分数** (30%): 正面朝向的目标威胁更高
- **朝向分数** (20%): 背面/侧面更容易击杀 (可配置)
- **大小分数** (10%): 目标越大越明显
- **置信度分数** (5%): 检测置信度

支持策略模式：综合权重、最近距离、最高威胁、最优朝向

### 2.4 压枪补偿系统
- **固定模式**: 预定义每种武器的后坐力模式，按发数施加反向补偿
- **自适应模式**: 根据命中反馈动态调整补偿强度
- **混合模式**: 固定模式为基础 + 自适应微调
- **武器库**: 步枪(rifle)、冲锋枪(smg)、机枪(lmg)、默认(default)
- **枪口回降**: 停止开火后模拟自然回落
- **配置参数**: 每发垂直偏移、水平漂移、最大补偿量、回降速度

### 2.5 准星校准系统
- **手动校准**: WASD微调偏移，+/-调整灵敏度，Enter保存
- **自动校准**: OpenCV检测画面中的准星标记，自动计算偏移
- **8方向校准矩阵**: 支持不同方向的灵敏度微调
- **实时预览**: 校准偏移可视化指示

### 2.6 ESP32-S3 HID桥接
- **二进制通信协议**: 高效帧格式 (包头+版本+指令+数据+校验)
- **多种连接方式**: UART串口、USB-CDC、UDP(WiFi)
- **心跳检测**: 自动重连、超时检测
- **指令队列**: 有界队列防溢出，旧指令丢弃
- **批量发送**: 合并连续移动指令减少通信开销

---

## 三、硬件要求

### Jetson AGX Xavier (推理端)
| 组件 | 规格 | 说明 |
|------|------|------|
| **主控** | Jetson AGX Xavier 32GB | Volta GPU 512 CUDA cores |
| **采集卡** | 低延迟HDMI采集卡 | MJPEG 1080p@60fps, V4L2 |
| **USB** | USB 3.0 | 连接采集卡 + ESP32-S3 |
| **存储** | 64GB+ | 模型+日志+校准数据 |

### ESP32-S3 (HID设备端)
| 组件 | 规格 | 说明 |
|------|------|------|
| **主控** | ESP32-S3 DevKitC | Dual-core Xtensa LX7 @ 240MHz |
| **USB** | USB-OTG (原生) | 模拟HID鼠标设备 |
| **UART** | 硬件UART | 接收Xavier指令 (可选USB-CDC) |
| **Flash** | 8MB+ | 固件存储 |

### 连接方式
**推荐方案 (USB-CDC)**:
```
Xavier USB口 ──USB线──▶ ESP32-S3 USB口
(识别为 /dev/ttyACM0)     (USB-OTG + HID复合设备)
```
**备选方案 (UART)**:
```
Xavier UART TX ──▶ ESP32-S3 RX (GPIO44)
Xavier UART RX ──▶ ESP32-S3 TX (GPIO43)
共地 (GND)
```

---

## 四、软件环境

### JetPack版本
- **JetPack 5.0+** (CUDA 11.4, TensorRT 8.4+, Python 3.8+)

### 基础依赖
```bash
pip install numpy opencv-python opencv-contrib-python pyserial
```

### ESP32开发环境
- Arduino IDE 2.x
- ESP32 Board Package 2.0.14+
- Board: "ESP32S3 Dev Module"
- USB Mode: "USB-OTG (TinyUSB)"
- USB CDC On Boot: Enabled

---

## 五、项目结构

```
rtmo_aimbot_enhanced/
├── main.py                          # 增强版主程序入口 (支持双后端推理)
├── src/
│   ├── __init__.py
│   ├── config.py                    # 全局配置 (含PyTorch/TensorRT双后端配置)
│   ├── pytorch_inference.py         # PyTorch原生推理引擎 (新增)
│   ├── tensorrt_wrapper.py          # TensorRT推理引擎
│   ├── rtmo_decoder.py              # RTMO解码 + 增强可视化
│   ├── aiming_engine.py             # 瞄准引擎增强版
│   ├── body_orientation.py          # 身体/枪口朝向估算
│   ├── recoil_compensator.py        # 压枪补偿系统
│   ├── crosshair_calibrator.py      # 准星校准系统
│   ├── esp32_bridge.py              # ESP32-S3通信桥接
│   ├── mouse_hid.py                 # 鼠标控制 (ESP32模式)
│   ├── video_capture.py             # 视频采集
│   └── utils.py                     # 工具函数
├── esp32_firmware/
│   └── esp32_hid_mouse/
│       └── esp32_hid_mouse.ino      # ESP32-S3 Arduino固件
├── scripts/
│   ├── export_rtmo_onnx.py          # ONNX导出
│   ├── onnx2trt.py                  # TensorRT转换
│   └── calibrate_mouse.py           # 鼠标校准
├── models/
│   ├── rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth   # PyTorch权重
│   └── rtmo_l_640x640.trt           # TensorRT引擎 (可选)
├── configs/
│   ├── rtmo-l_16xb16-600e_coco-640x640.py   # MMPose配置文件
│   └── calibration_matrix.json      # 校准数据存储
├── requirements.txt
└── README.md                        # 本文档
```

---

## 六、详细安装教程

### 步骤1: 准备 Jetson AGX Xavier 环境

#### 1.1 系统基础配置
```bash
# 更新系统
sudo apt-get update && sudo apt-get upgrade -y

# 安装基础工具
sudo apt-get install -y python3-pip python3-dev python3-opencv \
    libopencv-dev v4l-utils

# 创建项目目录
mkdir -p ~/rtmo_aimbot_enhanced && cd ~/rtmo_aimbot_enhanced
```

#### 1.2 安装 PyTorch (Jetson ARM64 专用版本)
```bash
# 查看当前 JetPack 版本
head -n 1 /etc/nv_tegra_release

# ===== JetPack 5.1.2 (CUDA 11.4) =====
# 下载 PyTorch 2.1.0 (NVIDIA 官方编译版)
wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/ \
    torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# 安装
pip3 install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl

# 验证安装
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); \
    print(f'CUDA available: {torch.cuda.is_available()}'); \
    print(f'CUDA version: {torch.version.cuda}')"

# 预期输出:
# PyTorch 2.1.0a0+41361538
# CUDA available: True
# CUDA version: 11.4
```

**常见问题排查:**
```bash
# Q: pip 安装 PyTorch 时提示 "not a supported wheel on this platform"
# A: 检查 Python 版本和平台架构
python3 --version  # 应为 3.8.x
uname -m           # 应为 aarch64

# Q: 找不到对应 JetPack 版本的 PyTorch
# A: 查看 NVIDIA 官方 PyTorch for Jetson 页面:
#    https://forums.developer.nvidia.com/t/\
#         pytorch-for-jetson/72048
#    根据 JetPack 版本选择对应 wheel 文件

# Q: 导入 torch 报错 "libcudnn.so.8: cannot open"
# A: JetPack 未正确安装，重新刷写 JetPack 镜像
sudo apt-get install --reinstall nvidia-jetpack
```

#### 1.3 安装 MMCV (MMPose 底层依赖)
```bash
# 安装前置依赖
sudo apt-get install -y libopenmpi-dev libopenblas-dev libomp-dev

# 方式1: 预编译 wheel (推荐, 较快)
pip install openmim
mim install mmcv==2.1.0

# 方式2: 从源码编译 (如果预编译版不兼容)
pip install mmengine
pip install -U openmim
mim install "mmcv>=2.0.0"

# 验证安装
python3 -c "import mmcv; print(f'MMCV {mmcv.__version__}')"
```

#### 1.4 安装 MMPose 和 MMEngine
```bash
# 安装 MMEngine
pip install mmengine>=0.7.0

# 安装 MMPose (不包含模型库)
pip install mmpose>=1.3.0

# 如果需要完整模型库 (下载配置文件和工具)
# git clone https://github.com/open-mmlab/mmpose.git
# cd mmpose && pip install -e .

# 验证安装
python3 -c "from mmpose.apis import init_model; print('MMPose 安装成功')"
```

**常见问题排查:**
```bash
# Q: 安装 mmpose 时依赖冲突
# A: 先安装 mmengine，再安装 mmpose
pip install --upgrade pip
pip install mmengine
pip install mmpose --no-deps  # 跳过依赖检查

# Q: 导入 mmpose 报错 "No module named 'mmcv._ext'"
# A: MMCV 未正确编译，重新安装
pip uninstall mmcv mmcv-lite -y
mim install mmcv==2.1.0

# Q: 运行时报错 "RTMO is not in the mmpose::model registry"
# A: MMPose 版本过旧，需要 1.3.0+
pip install --upgrade mmpose>=1.3.0
```

#### 1.5 安装其他 Python 依赖
```bash
# 基础依赖
pip install numpy opencv-python opencv-contrib-python pyserial

# TensorRT 支持 (可选, 仅使用 --backend tensorrt 时需要)
# Jetson 已预装 TensorRT，只需安装 pycuda
# pip install pycuda
```

---

### 步骤2: 下载 RTMO-l 模型文件

#### 2.1 创建目录结构
```bash
cd ~/rtmo_aimbot_enhanced
mkdir -p models configs
```

#### 2.2 下载 PyTorch 权重文件 (.pth)
```bash
# 从 MMPose Model Zoo 下载 RTMO-l COCO 640x640 权重
cd models

# 方式1: 直接下载
wget https://download.openmmlab.com/mmpose/v1/projects/rtmo/ \
    rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth

# 方式2: 使用 MMPose 工具下载
python3 -c "
from mmpose.utils import get_model_file
url = 'https://download.openmmlab.com/mmpose/v1/projects/rtmo/\
rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth'
import urllib.request
urllib.request.urlretrieve(url, 'rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth')
print('下载完成')
"

# 验证文件完整性 (检查文件大小, 约 180MB)
ls -lh rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth
# 预期: ~180MB
```

#### 2.3 下载 MMPose 配置文件
```bash
cd ~/rtmo_aimbot_enhanced/configs

# 方式1: 从 MMPose GitHub 仓库下载
wget https://raw.githubusercontent.com/open-mmlab/mmpose/main/ \
    projects/rtmo/rtmo-l_16xb16-600e_coco-640x640.py

# 方式2: 如果安装了完整 MMPose 仓库，直接复制
# cp /path/to/mmpose/projects/rtmo/rtmo-l_16xb16-600e_coco-640x640.py .

# 方式3: 手动创建最小配置文件 (如果上述方式不可用)
# 配置文件内容较长，建议直接从 GitHub 下载
```

#### 2.4 验证模型文件
```bash
cd ~/rtmo_aimbot_enhanced

# 运行验证脚本
python3 -c "
import torch
import os

pth_path = 'models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth'
config_path = 'configs/rtmo-l_16xb16-600e_coco-640x640.py'

# 检查文件存在
assert os.path.exists(pth_path), f'权重文件不存在: {pth_path}'
assert os.path.exists(config_path), f'配置文件不存在: {config_path}'

# 加载检查点
checkpoint = torch.load(pth_path, map_location='cpu')
print('检查点键:', list(checkpoint.keys()))

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f'状态字典键数量: {len(state_dict.keys())}')
    print('部分键名示例:')
    for i, k in enumerate(list(state_dict.keys())[:5]):
        print(f'  {k}: {state_dict[k].shape}')
    print('\\n模型文件验证通过!')
else:
    print('警告: 检查点中没有 state_dict 键')
    print('可用键:', list(checkpoint.keys()))
"
```

---

### 步骤3: 烧录ESP32-S3固件

1. 使用Arduino IDE打开 `esp32_firmware/esp32_hid_mouse/esp32_hid_mouse.ino`
2. 选择 Board: "ESP32S3 Dev Module"
3. USB Mode: "USB-OTG (TinyUSB)"
4. 上传固件到ESP32-S3
5. 确认ESP32-S3被识别为USB鼠标设备 (在电脑上测试)

---

### 步骤4: 连接硬件

**USB-CDC方案 (推荐)**:
```bash
# 使用USB线连接Xavier和ESP32-S3
# ESP32-S3会在Xavier上显示为 /dev/ttyACM0
ls /dev/ttyACM*
```

**UART方案**:
```
Xavier UART1_TX (Pin 8)  → ESP32-S3 RX (GPIO44)
Xavier UART1_RX (Pin 10) → ESP32-S3 TX (GPIO43)
Xavier GND               → ESP32-S3 GND
```

---

### 步骤5: 配置权限

```bash
# 1. 加载uinput模块 (仅local模式需要)
sudo modprobe uinput

# 2. 串口权限
sudo chmod 666 /dev/ttyACM0

# 3. 创建udev规则 (持久化)
sudo tee /etc/udev/rules.d/99-aimbot.rules << 'EOF'
KERNEL=="ttyACM[0-9]*", MODE="0666", GROUP="dialout"
KERNEL=="uinput", MODE="0666"
KERNEL=="video[0-9]*", MODE="0666"
EOF
sudo udevadm control --reload-rules
```

---

### 步骤6: 运行程序

#### 6.1 PyTorch 原生推理 (默认, 推荐首次使用)
```bash
cd ~/rtmo_aimbot_enhanced

# 虚拟模式测试 (无实际硬件, 仅测试推理)
sudo python3 main.py --debug --dummy-esp32 --dummy-mouse

# 生产模式 (使用 ESP32-S3)
sudo python3 main.py \
    --backend pytorch \
    --pytorch-config configs/rtmo-l_16xb16-600e_coco-640x640.py \
    --pytorch-checkpoint models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth \
    --mouse-mode esp32 \
    --serial-port /dev/ttyACM0 \
    --debug

# 简写 (使用默认路径)
sudo python3 main.py --mouse-mode esp32 --serial-port /dev/ttyACM0 --debug
```

#### 6.2 TensorRT 推理 (需先转换)
```bash
# 步骤1: 导出 ONNX (在 x86 开发机上执行较方便)
python3 scripts/export_rtmo_onnx.py \
    --config configs/rtmo-l_16xb16-600e_coco-640x640.py \
    --checkpoint models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth \
    --output models/

# 步骤2: 转换为 TensorRT (在 Jetson 上执行)
python3 scripts/onnx2trt.py \
    --onnx models/rtmo_l_640x640.onnx \
    --output models/rtmo_l_640x640.trt \
    --fp16 --workspace 2048

# 步骤3: 运行
sudo python3 main.py \
    --backend tensorrt \
    --engine models/rtmo_l_640x640.trt \
    --mouse-mode esp32 \
    --serial-port /dev/ttyACM0 \
    --debug
```

---

## 七、键盘控制说明

| 按键 | 功能 |
|------|------|
| **F1** | 切换武器配置 (rifle/smg/lmg/default) |
| **F2** | 开关压枪补偿 |
| **F3** | 进入手动准星校准模式 |
| **F4** | 触发自动准星校准 |
| **F5** | 切换目标选择策略 |
| **+ / =** | 增加灵敏度 |
| **- / _** | 降低灵敏度 |
| **1** | 瞄准头部 |
| **2** | 瞄准躯干 |
| **W/A/S/D** | 校准模式下微调偏移 |
| **Enter** | 校准模式下保存并退出 |
| **ESC** | 退出校准模式 / 退出程序 |
| **Q** | 退出程序 |

---

## 八、配置调优

### 8.1 推理后端切换
```python
# src/config.py - MODEL_CFG
MODEL_CFG.inference_backend = "pytorch"   # PyTorch 原生推理 (默认)
MODEL_CFG.inference_backend = "tensorrt"  # TensorRT 推理

# PyTorch 配置
MODEL_CFG.pytorch_config_path = "configs/rtmo-l_16xb16-600e_coco-640x640.py"
MODEL_CFG.pytorch_checkpoint_path = "models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth"
MODEL_CFG.pytorch_device = "cuda:0"       # 推理设备
MODEL_CFG.pytorch_fp16 = True             # FP16 半精度 (加速推理)

# TensorRT 配置
MODEL_CFG.engine_path = "models/rtmo_l_640x640.trt"
```

### 8.2 身体朝向估算
```python
# src/config.py - BODY_ORI_CFG
BODY_ORI_CFG.enabled = True
BODY_ORI_CFG.facing_front_thresh = 30.0   # 正面判定角度阈值
BODY_ORI_CFG.shoulder_hip_ratio_front = 1.3  # 正面向宽高比
BODY_ORI_CFG.history_smooth_frames = 3    # 平滑帧数
```

### 8.3 目标优先级
```python
# src/config.py - TARGET_PRIO_CFG
TARGET_PRIO_CFG.strategy = "composite"  # 综合权重策略
TARGET_PRIO_CFG.w_distance = 0.35       # 距离权重
TARGET_PRIO_CFG.w_threat = 0.30         # 威胁度权重
TARGET_PRIO_CFG.w_orientation = 0.20    # 朝向权重
TARGET_PRIO_CFG.prioritize_back = True  # 优先击杀背身目标
```

### 8.4 压枪补偿
```python
# src/config.py - RECOIL_CFG
RECOIL_CFG.enabled = True
RECOIL_CFG.mode = "adaptive"  # adaptive/pattern/hybrid
RECOIL_CFG.current_weapon = "rifle"

# 自定义武器配置
RECOIL_CFG.weapon_profiles["my_rifle"] = {
    "vertical_per_shot": 2.0,    # 每发垂直上升量 (像素)
    "horizontal_drift": 0.4,     # 水平漂移范围
    "max_compensation": 20.0,    # 最大单帧补偿
    "recovery_rate": 0.35,       # 枪口回降速度
    "bullets_per_pattern": 30,   # 弹匣容量
}
```

### 8.5 准星校准
```python
# src/config.py - CALIB_CFG
CALIB_CFG.enabled = True
CALIB_CFG.mode = "manual"           # manual/auto/semi_auto
CALIB_CFG.calibration_distance = 100  # 校准步长 (像素)
```

### 8.6 ESP32通信
```python
# src/config.py - ESP32_CFG
ESP32_CFG.mode = "serial"           # serial/udp/usb_cdc
ESP32_CFG.serial_port = "/dev/ttyACM0"
ESP32_CFG.serial_baudrate = 921600
ESP32_CFG.use_binary_protocol = True  # True=二进制, False=JSON
ESP32_CFG.auto_reconnect = True
```

---

## 九、通信协议详解

### 9.1 二进制协议 (推荐)

**发送帧格式 (Xavier → ESP32)**:
```
[0xAA][0x55][VERSION][CMD][LEN_LO][LEN_HI][DATA...][CHECKSUM]
  2B     1B      1B     2B          NB         1B
```

**响应帧格式 (ESP32 → Xavier)**:
```
[0xBB][0x66][VERSION][RESP][STATUS][LEN_LO][LEN_HI][DATA...][CHECKSUM]
  2B     1B      1B      1B       2B          NB         1B
```

**校验和**: 包头至数据末尾所有字节累加和的低8位

### 9.2 指令类型

| 指令 | 代码 | 数据 | 说明 |
|------|------|------|------|
| 鼠标移动 | 0x01 | dx(int16)+dy(int16)+buttons(uint8) | 相对移动 |
| 鼠标按键 | 0x02 | button(uint8)+state(uint8) | 按下/释放 |
| 滚轮 | 0x03 | vertical(int8)+horizontal(int8) | 滚轮滚动 |
| 综合 | 0x04 | dx+dy+buttons+wheelV | 移动+按键+滚轮 |
| 心跳 | 0x05 | 无 | 保活检测 |
| 状态查询 | 0x06 | 无 | 查询ESP32状态 |
| 配置 | 0x07 | key(uint8)+value(int16) | 参数配置 |

### 9.3 JSON协议 (调试)

设置 `ESP32_CFG.use_binary_protocol = False` 启用：
```json
{"cmd": "move", "dx": 10, "dy": -5, "btn": 0, "ts": 1234567890.123}
```

---

## 十、性能指标

| 指标 | 原版 | PyTorch (.pth) | TensorRT (.trt) | 说明 |
|------|------|----------------|-----------------|------|
| **端到端延迟** | ~50-55ms | ~35-45ms | ~25-35ms | 流水线优化 |
| **推理帧率** | 28-35 FPS | 32-38 FPS | 40-50 FPS | 始终处理最新帧 |
| **推理时间** | ~40ms | ~25-35ms | ~15-25ms | 模型前向传播 |
| **鼠标回报等效** | 125 Hz | 1000 Hz | 1000 Hz | 事件驱动+批量合并 |
| **目标选择** | 距离最近 | 综合权重 | 综合权重 | 朝向感知+威胁评估 |
| **压枪补偿** | 无 | 3种模式 | 3种模式 | 固定/自适应/混合 |
| **准星校准** | 无 | 自动+手动 | 自动+手动 | 8方向校准矩阵 |
| **身体朝向** | 无 | 4方向+8细分 | 4方向+8细分 | 肩线向量法 |

### PyTorch FP16 vs FP32 性能对比

| 模式 | 推理时间 | GPU显存 | 精度 |
|------|----------|---------|------|
| FP32 | ~35-45ms | ~850MB | 基准 |
| FP16 | ~25-35ms | ~550MB | 几乎无损 |

建议: Jetson AGX Xavier 默认启用 FP16，可提升约 30% 推理速度。

---

## 十一、调试指南

### 11.1 推理引擎调试

#### 检查 PyTorch 推理环境
```bash
# 运行完整环境检查脚本
python3 -c "
import sys
print(f'Python: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        # 测试简单推理
        x = torch.randn(1, 3, 640, 640).cuda()
        print('GPU 推理测试: OK')
except ImportError:
    print('PyTorch: 未安装')

try:
    import mmcv
    print(f'MMCV: {mmcv.__version__}')
except ImportError:
    print('MMCV: 未安装')

try:
    import mmpose
    print(f'MMPose: {mmpose.__version__}')
except ImportError:
    print('MMPose: 未安装')

try:
    import mmengine
    print(f'MMEngine: {mmengine.__version__}')
except ImportError:
    print('MMEngine: 未安装')

try:
    import cv2
    print(f'OpenCV: {cv2.__version__}')
except ImportError:
    print('OpenCV: 未安装')
"
```

#### 测试模型加载
```bash
# 独立测试模型加载和推理
python3 -c "
import torch
import time
import numpy as np
from mmpose.apis import init_model

config_path = 'configs/rtmo-l_16xb16-600e_coco-640x640.py'
checkpoint_path = 'models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth'

print('正在加载模型...')
t0 = time.time()
model = init_model(config_path, checkpoint_path, device='cuda:0')
load_time = time.time() - t0
print(f'模型加载完成: {load_time:.2f}s')

# 测试推理
print('\\n测试推理...')
dummy_input = torch.zeros(1, 3, 640, 640).cuda()

# 预热
for _ in range(3):
    _ = model.test_step(dict(inputs=dummy_input, data_samples=None))
torch.cuda.synchronize()

# 正式测试
times = []
for _ in range(10):
    t0 = time.time()
    result = model.test_step(dict(inputs=dummy_input, data_samples=None))
    torch.cuda.synchronize()
    times.append(time.time() - t0)

avg_time = np.mean(times) * 1000
print(f'平均推理时间: {avg_time:.2f}ms')
print(f'理论帧率: {1000/avg_time:.1f} FPS')

# 检查结果格式
if isinstance(result, list) and len(result) > 0:
    pred = result[0]
    if hasattr(pred, 'pred_instances'):
        inst = pred.pred_instances
        print(f'\\n检测结果:')
        print(f'  检测到人数: {len(inst)}')
        if len(inst) > 0:
            print(f'  bbox 形状: {inst.bboxes.shape}')
            print(f'  keypoints 形状: {inst.keypoints.shape}')
print('\\n模型测试通过!')
"
```

#### PyTorch vs TensorRT 精度对比
```bash
# 运行两种后端对比测试
python3 -c "
import numpy as np
import torch

# 准备相同输入
np.random.seed(42)
test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

print('===== PyTorch 推理 =====')
from src.pytorch_inference import PytorchInferenceEngine
from src.config import MODEL_CFG

pt_engine = PytorchInferenceEngine(
    MODEL_CFG.pytorch_config_path,
    MODEL_CFG.pytorch_checkpoint_path,
    fp16=False  # FP32 对比
)
pt_outputs = pt_engine.infer(test_input)
print(f'PyTorch 输出: dets={pt_outputs[0].shape}, kpts={pt_outputs[1].shape}')

# print('===== TensorRT 推理 =====')
# from src.tensorrt_wrapper import TrtInferenceEngine
# trt_engine = TrtInferenceEngine(MODEL_CFG.engine_path)
# trt_outputs = trt_engine.infer(test_input)
# print(f'TensorRT 输出: dets={trt_outputs[0].shape}, kpts={trt_outputs[1].shape}')

print('\\n输出格式一致，可以共用同一个 decoder!')
"
```

### 11.2 模型检测调试

#### 检查模型文件完整性
```bash
# 检查 .pth 文件
python3 -c "
import torch
import os

pth_path = 'models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth'

if not os.path.exists(pth_path):
    print(f'错误: 文件不存在: {pth_path}')
    exit(1)

# 检查文件大小
size_mb = os.path.getsize(pth_path) / (1024 * 1024)
print(f'文件大小: {size_mb:.1f} MB')
if size_mb < 100:
    print('警告: 文件过小，可能下载不完整!')

# 尝试加载
try:
    ckpt = torch.load(pth_path, map_location='cpu')
    print('文件加载成功')
    
    if 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        print(f'state_dict 键数量: {len(sd)}')
        
        # 检查关键层是否存在
        key_layers = ['backbone', 'neck', 'head']
        for layer in key_layers:
            matches = [k for k in sd.keys() if k.startswith(layer)]
            print(f'  {layer} 层参数数量: {len(matches)}')
    else:
        print('警告: 未找到 state_dict')
        print('可用键:', list(ckpt.keys()))
        
except Exception as e:
    print(f'加载失败: {e}')
    print('文件可能已损坏，请重新下载')
"
```

#### 配置文件验证
```bash
# 验证 MMPose 配置文件
python3 -c "
from mmengine.config import Config
import os

config_path = 'configs/rtmo-l_16xb16-600e_coco-640x640.py'

if not os.path.exists(config_path):
    print(f'错误: 配置文件不存在: {config_path}')
    print('请从 MMPose 官方仓库下载:')
    print('  https://github.com/open-mmlab/mmpose/tree/main/projects/rtmo')
    exit(1)

try:
    cfg = Config.fromfile(config_path)
    print('配置文件解析成功')
    print(f'模型类型: {cfg.model.type}')
    print(f'主干网络: {cfg.model.backbone.type}')
    print(f'输入尺寸: {cfg.model.data_preprocessor.get(\"input_size\", \"unknown\")}')
except Exception as e:
    print(f'配置文件解析失败: {e}')
    print('可能原因:')
    print('  1. 配置文件格式错误')
    print('  2. 缺少基础配置 (_base_)')
    print('  3. MMPose 版本不兼容')
"
```

#### 单帧推理测试 (快速验证)
```bash
# 使用单张图片测试完整推理流程
python3 -c "
import cv2
import numpy as np
import time

from src.pytorch_inference import PytorchInferenceEngine
from src.tensorrt_wrapper import preprocess_image
from src.rtmo_decoder import RTMODecoder
from src.config import MODEL_CFG

# 加载引擎
print('加载推理引擎...')
engine = PytorchInferenceEngine(
    MODEL_CFG.pytorch_config_path,
    MODEL_CFG.pytorch_checkpoint_path,
    fp16=True
)

# 创建解码器
decoder = RTMODecoder(
    conf_thresh=MODEL_CFG.conf_thresh,
    nms_thresh=MODEL_CFG.nms_thresh
)

# 读取测试图片 (或使用采集卡)
print('\\n读取测试帧...')
cap = cv2.VideoCapture('/dev/video0')
ret, frame = cap.read()
cap.release()

if not ret:
    print('采集卡读取失败，使用随机图片测试')
    frame = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)
else:
    print(f'读取帧: {frame.shape}')

# 预处理
preprocessed, scale, pad_offset = preprocess_image(
    frame,
    target_size=(MODEL_CFG.input_width, MODEL_CFG.input_height),
    mean=MODEL_CFG.mean,
    std=MODEL_CFG.std
)

# 推理
t0 = time.time()
outputs = engine.infer(preprocessed)
infer_ms = (time.time() - t0) * 1000

# 解码
t0 = time.time()
persons = decoder.decode(outputs, scale, pad_offset, frame.shape[:2])
decode_ms = (time.time() - t0) * 1000

print(f'\\n推理时间: {infer_ms:.1f}ms')
print(f'解码时间: {decode_ms:.1f}ms')
print(f'检测到人数: {len(persons)}')

for i, p in enumerate(persons[:3]):
    print(f'  人物{i+1}: score={p[\"score\"]:.3f}, '
          f'bbox={p[\"bbox\"].astype(int).tolist()}')

engine.release()
print('\\n测试完成!')
"
```

### 11.3 HID鼠标校准调试

#### 检查 ESP32-S3 连接
```bash
# 检查设备识别
lsusb | grep -i "ESP\|USB Serial"
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null

# 检查串口权限
ls -la /dev/ttyACM0
# 应为: crw-rw-rw- (666权限)

# 如果没有权限，执行:
sudo chmod 666 /dev/ttyACM0
sudo usermod -a -G dialout $USER

# 查看内核日志
dmesg | tail -30 | grep -i "usb\|acm\|esp"

# 测试串口通信
python3 -c "
import serial
import time

port = '/dev/ttyACM0'
try:
    s = serial.Serial(port, 921600, timeout=1)
    print(f'串口打开成功: {port}')
    
    # 发送心跳测试
    heartbeat = b'\\xaa\\x55\\x01\\x05\\x00\\x00\\xab'
    s.write(heartbeat)
    print('心跳包已发送')
    
    # 等待响应
    time.sleep(0.5)
    if s.in_waiting > 0:
        resp = s.read(s.in_waiting)
        print(f'收到响应: {resp.hex()}')
    else:
        print('未收到响应 (可能 ESP32 未就绪)')
    
    s.close()
except Exception as e:
    print(f'串口错误: {e}')
"
```

#### 鼠标移动单位校准
```bash
# 运行鼠标校准脚本
python3 scripts/calibrate_mouse.py

# 或使用自动校准
python3 scripts/calibrate_mouse.py --auto --distance 200 --samples 5
```

**校准原理说明:**
1. 系统发送 100 单位的鼠标移动指令
2. 观察游戏中准星实际移动了多少像素
3. 计算比例系数: `sensitivity = 发送值 / 实际移动像素`
4. 例如: 发送 100，实际移动 80 像素 -> sensitivity = 1.25

**常见校准值参考:**
| 游戏 | 灵敏度系数 | 游戏内灵敏度设置 |
|------|-----------|----------------|
| CS2 | 1.0 - 1.5 | 依个人习惯 |
| Valorant | 0.3 - 0.5 | 依个人习惯 |
| Apex | 1.0 - 2.0 | 依个人习惯 |

#### ESP32 固件调试模式
```bash
# 在 Arduino IDE 中开启串口监视器
# 波特率: 921600

# 预期输出:
# ========================================
# ESP32-S3 HID Mouse Device
# Firmware v1.0
# ========================================
# Waiting for commands...

# 如果看不到输出:
# 1. 检查波特率设置是否正确
# 2. 确认 USB 线支持数据传输 (不是充电线)
# 3. 尝试更换 USB 口
# 4. 在 Arduino IDE 中重新烧录固件
```

### 11.4 常见问题排查 (Q&A)

#### Q1: PyTorch 推理报错 "CUDA out of memory"
```bash
# 解决方案:
# 1. 启用 FP16 半精度推理 (减少约40%显存)
#    已在 config.py 中默认启用: pytorch_fp16 = True

# 2. 降低输入分辨率
#    修改 config.py: input_width = 512, input_height = 512

# 3. 释放其他程序的 GPU 内存
sudo fuser -v /dev/nvidia*  # 查看占用 GPU 的进程

# 4. 重启 Jetson 释放显存
sudo reboot
```

#### Q2: 模型加载报错 "No module named 'mmpose'"
```bash
# 完整安装步骤:
# 1. 安装 mmengine
pip install mmengine>=0.7.0

# 2. 安装 mmcv (Jetson ARM64 专用)
pip install openmim
mim install "mmcv>=2.0.0"

# 3. 安装 mmpose
pip install mmpose>=1.3.0

# 4. 验证
python3 -c "from mmpose.apis import init_model; print('OK')"
```

#### Q3: PyTorch 推理比 TensorRT 慢很多
```bash
# 正常现象，但可以通过以下方式优化:

# 1. 确认 FP16 已启用
python3 -c "from src.config import MODEL_CFG; print(MODEL_CFG.pytorch_fp16)"
# 应为 True

# 2. 使用 torch.compile (PyTorch 2.0+)
# 在 config.py 中设置: use_cuda_graph = True

# 3. 转换为 TensorRT 以获得最佳性能
python3 scripts/onnx2trt.py \
    --onnx models/rtmo_l_640x640.onnx \
    --output models/rtmo_l_640x640.trt \
    --fp16

# 然后使用 TensorRT 后端运行
python3 main.py --backend tensorrt --engine models/rtmo_l_640x640.trt
```

#### Q4: 检测结果为空或置信度很低
```bash
# 1. 检查置信度阈值
python3 -c "from src.config import MODEL_CFG; print(f'conf_thresh={MODEL_CFG.conf_thresh}')"
# 可以适当降低 (如 0.2) 进行测试

# 2. 检查视频输入
v4l2-ctl -d /dev/video0 --all  # 查看采集卡参数
v4l2-ctl -d /dev/video0 --list-formats-ext  # 查看支持的分辨率

# 3. 测试采集卡
ffplay -f v4l2 -i /dev/video0 -video_size 1920x1080 -framerate 60

# 4. 检查预处理参数
python3 -c "
from src.config import MODEL_CFG
print(f'mean={MODEL_CFG.mean}')
print(f'std={MODEL_CFG.std}')
print(f'input_size=({MODEL_CFG.input_width}, {MODEL_CFG.input_height})')
# 确保这些值与 MMPose 训练时使用的值一致
"
```

#### Q5: ESP32-S3 无法连接
```bash
# 检查设备识别
lsusb | grep -i esp32
ls /dev/ttyACM*

# 检查权限
sudo chmod 666 /dev/ttyACM0
sudo usermod -a -G dialout $USER

# 查看内核日志
dmesg | tail -20

# 如果设备未识别:
# 1. 检查 USB 线是否支持数据传输
# 2. 按住 BOOT 按钮再按 RESET 进入下载模式
# 3. 在 Arduino IDE 中重新选择端口
```

#### Q6: HID 鼠标无响应
- 确认ESP32-S3被识别为USB鼠标 (在其他电脑上测试)
- 检查Arduino IDE中USB Mode设置为"USB-OTG (TinyUSB)"
- 确认固件中的 `USB.begin()` 和 `Mouse.begin()` 正确调用

#### Q7: 串口通信丢包
- 降低波特率: `ESP32_CFG.serial_baudrate = 115200`
- 改用USB-CDC模式 (更稳定)
- 检查接线 (UART方案需要交叉连接TX/RX)

#### Q8: 身体朝向判断不准
- 调整 `facing_front_thresh` 角度阈值
- 确保人体肩部关键点可见
- 增加 `history_smooth_frames` 平滑帧数

#### Q9: 压枪补偿过度/不足
- 调整武器配置中的 `vertical_per_shot`
- 切换压枪模式: `RECOIL_CFG.mode = "adaptive"`
- 在游戏中实测后调整灵敏度

#### Q10: PyTorch 与 TensorRT 输出结果不一致
```bash
# 正常现象: TensorRT FP16 可能有微小精度损失
# 如果差异很大:

# 1. 检查 TensorRT 转换参数
#    确保 --fp16 标志一致

# 2. 检查预处理是否完全相同
#    PyTorch 和 TensorRT 使用相同的 preprocess_image 函数

# 3. 重新导出 ONNX 并转换
python3 scripts/export_rtmo_onnx.py \
    --config configs/rtmo-l_16xb16-600e_coco-640x640.py \
    --checkpoint models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth \
    --output models/

python3 scripts/onnx2trt.py \
    --onnx models/rtmo_l_640x640.onnx \
    --output models/rtmo_l_640x640.trt \
    --fp16
```

---

## 十二、安全声明

> **本系统仅通过HDMI视频环出 + HID鼠标模拟与游戏交互**
> - 不读取游戏内存
> - 不修改游戏文件
> - 不注入任何代码
> - 不拦截网络封包
> 
> 本项目的目的是研究计算机视觉在实时场景下的边缘部署技术、
> 人体姿态估计算法应用、嵌入式系统通信协议等技术领域。
> 请遵守各游戏平台的服务条款与社区准则。
> 禁止将该项目用于违法或者违规商业用途，违者后果自负与开发团队无关

---

## 十三、参考文献

1. [RTMO: Real-Time Multi-person pose estimation Optimized](https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmo)
2. [MMPose Documentation](https://mmpose.readthedocs.io/)
3. [MMPose - Inference with existing models](https://mmpose.readthedocs.io/en/latest/user_guides/inference.html)
4. [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
5. [ESP32-S3 USB OTG Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/peripherals/usb_device.html)
6. [Arduino USBHIDMouse Library](https://github.com/espressif/arduino-esp32/tree/master/libraries/USB/src)
7. [PyTorch for Jetson - NVIDIA Developer Forums](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
8. [MMCV Installation Guide](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)
9. [Secrets of Gosu: Understanding physical combat skills of professional players in first-person shooters](https://dl.acm.org/doi/abs/10.1145/3411764.3445217)
10. [Application of Low-Rank Approximation via SVD for Detecting Synthetic Recoil Control in Valorant](https://informatika.stei.itb.ac.id/)
11. [Human Body Orientation from 2D Images (SAE 2021)](https://saemobilus.sae.org/)

---

## 十四、版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-26 | 初始版本 (基础多线程流水线) |
| v2.0 | 2026-05-30 | 增强版 (身体朝向/压枪/校准/ESP32桥接) |
| v2.1 | 2026-06-01 | 新增 PyTorch 原生推理引擎 (支持 .pth 权重文件, FP16, 双后端切换) |

---

**开发团队**: niko智能-SYJ
**许可证**: Apache License (仅供技术研究使用)
