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
│  │ V4L2采集   │  │TensorRT推理│  │ 可视化/校准UI     │ │
│  │            │  │RTMO解码   │  │ 键盘交互控制      │ │
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

### 2.1 身体/枪口朝向估算
- **肩线向量法**: 通过左右肩关键点构建参考向量，分析肩线与水平面的夹角
- **髋线验证法**: 使用左右髋关键点辅助验证朝向判断
- **宽高比分析法**: 肩宽/髋宽比例区分正面/侧面/背面
- **8方向分类**: 正面、背面、左侧面、右侧面 + 4个斜向
- **时间域平滑**: 3帧历史数据平滑，减少抖动

### 2.2 多目标优先级选择策略
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

### 2.3 压枪补偿系统
- **固定模式**: 预定义每种武器的后坐力模式，按发数施加反向补偿
- **自适应模式**: 根据命中反馈动态调整补偿强度
- **混合模式**: 固定模式为基础 + 自适应微调
- **武器库**: 步枪(rifle)、冲锋枪(smg)、机枪(lmg)、默认(default)
- **枪口回降**: 停止开火后模拟自然回落
- **配置参数**: 每发垂直偏移、水平漂移、最大补偿量、回降速度

### 2.4 准星校准系统
- **手动校准**: WASD微调偏移，+/-调整灵敏度，Enter保存
- **自动校准**: OpenCV检测画面中的准星标记，自动计算偏移
- **8方向校准矩阵**: 支持不同方向的灵敏度微调
- **实时预览**: 校准偏移可视化指示

### 2.5 ESP32-S3 HID桥接
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

### Python依赖
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
├── main.py                          # 增强版主程序入口
├── src/
│   ├── __init__.py
│   ├── config.py                    # 全局配置
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
│   └── rtmo_l_640x640.trt           # TensorRT引擎
├── configs/
│   └── calibration_matrix.json      # 校准数据存储
├── requirements.txt
└── README.md                        # 本文档
```

---

## 六、部署步骤

### 步骤1: 烧录ESP32-S3固件

1. 使用Arduino IDE打开 `esp32_firmware/esp32_hid_mouse/esp32_hid_mouse.ino`
2. 选择 Board: "ESP32S3 Dev Module"
3. USB Mode: "USB-OTG (TinyUSB)"
4. 上传固件到ESP32-S3
5. 确认ESP32-S3被识别为USB鼠标设备 (在电脑上测试)

### 步骤2: 连接硬件

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

### 步骤3: 配置权限

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

### 步骤4: 运行

```bash
# 虚拟模式测试 (无实际硬件)
sudo python3 main.py --debug --dummy-esp32 --dummy-mouse

# ESP32桥接模式 (生产环境)
sudo python3 main.py --mouse-mode esp32 --serial-port /dev/ttyACM0

# 带参数运行
sudo python3 main.py \
    --engine models/rtmo_l_640x640.trt \
    --conf 0.3 \
    --sensitivity 1.2 \
    --weapon rifle \
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

### 8.1 身体朝向估算
```python
# src/config.py - BODY_ORI_CFG
BODY_ORI_CFG.enabled = True
BODY_ORI_CFG.facing_front_thresh = 30.0   # 正面判定角度阈值
BODY_ORI_CFG.shoulder_hip_ratio_front = 1.3  # 正面向宽高比
BODY_ORI_CFG.history_smooth_frames = 3    # 平滑帧数
```

### 8.2 目标优先级
```python
# src/config.py - TARGET_PRIO_CFG
TARGET_PRIO_CFG.strategy = "composite"  # 综合权重策略
TARGET_PRIO_CFG.w_distance = 0.35       # 距离权重
TARGET_PRIO_CFG.w_threat = 0.30         # 威胁度权重
TARGET_PRIO_CFG.w_orientation = 0.20    # 朝向权重
TARGET_PRIO_CFG.prioritize_back = True  # 优先击杀背身目标
```

### 8.3 压枪补偿
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

### 8.4 准星校准
```python
# src/config.py - CALIB_CFG
CALIB_CFG.enabled = True
CALIB_CFG.mode = "manual"           # manual/auto/semi_auto
CALIB_CFG.calibration_distance = 100  # 校准步长 (像素)
```

### 8.5 ESP32通信
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

| 指标 | 原版 | 增强版 | 说明 |
|------|------|--------|------|
| **端到端延迟** | ~50-55ms | ~30-40ms | 流水线优化 |
| **推理帧率** | 18-25 FPS | 35-40 FPS | 始终处理最新帧 |
| **鼠标回报等效** | 125 Hz | 1000 Hz | 事件驱动+批量合并 |
| **目标选择** | 距离最近 | 综合权重 | 朝向感知+威胁评估 |
| **压枪补偿** | 无 | 3种模式 | 固定/自适应/混合 |
| **准星校准** | 无 | 自动+手动 | 8方向校准矩阵 |
| **身体朝向** | 无 | 4方向+8细分 | 肩线向量法 |

---

## 十一、故障排除

### Q1: ESP32-S3无法连接
```bash
# 检查设备识别
lsusb | grep -i esp32
ls /dev/ttyACM*

# 检查权限
sudo chmod 666 /dev/ttyACM0
sudo usermod -a -G dialout $USER

# 查看内核日志
dmesg | tail -20
```

### Q2: HID鼠标无响应
- 确认ESP32-S3被识别为USB鼠标 (在其他电脑上测试)
- 检查Arduino IDE中USB Mode设置为"USB-OTG (TinyUSB)"
- 确认固件中的 `USB.begin()` 和 `Mouse.begin()` 正确调用

### Q3: 串口通信丢包
- 降低波特率: `ESP32_CFG.serial_baudrate = 115200`
- 改用USB-CDC模式 (更稳定)
- 检查接线 (UART方案需要交叉连接TX/RX)

### Q4: 身体朝向判断不准
- 调整 `facing_front_thresh` 角度阈值
- 确保人体肩部关键点可见
- 增加 `history_smooth_frames` 平滑帧数

### Q5: 压枪补偿过度/不足
- 调整武器配置中的 `vertical_per_shot`
- 切换压枪模式: `RECOIL_CFG.mode = "adaptive"`
- 在游戏中实测后调整灵敏度

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
3. [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
4. [ESP32-S3 USB OTG Documentation](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/api-reference/peripherals/usb_device.html)
5. [Arduino USBHIDMouse Library](https://github.com/espressif/arduino-esp32/tree/master/libraries/USB/src)
6. [Secrets of Gosu: Understanding physical combat skills of professional players in first-person shooters](https://dl.acm.org/doi/abs/10.1145/3411764.3445217)
7. [Application of Low-Rank Approximation via SVD for Detecting Synthetic Recoil Control in Valorant](https://informatika.stei.itb.ac.id/)
8. [Human Body Orientation from 2D Images (SAE 2021)](https://saemobilus.sae.org/)

---

## 十四、版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| v1.0 | 2026-05-26 | 初始版本 (基础多线程流水线) |
| v2.0 | 2026-05-30 | 增强版 (身体朝向/压枪/校准/ESP32桥接) |

---

**开发团队**: niko智能-SYJ
**许可证**: Apache License (仅供技术研究使用)
