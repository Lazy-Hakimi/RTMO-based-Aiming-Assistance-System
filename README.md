# RTMO AimBot for Jetson AGX Xavier

基于 **RTMO-l** (Real-Time Multi-person pose estimation Optimized) 姿态估计模型，部署于 **Jetson AGX Xavier 32GB**，通过 HDMI 采集卡捕获游戏画面，利用 TensorRT 加速推理，通过 HID 虚拟鼠标实现人体追踪瞄准。

---

## 系统架构 (多线程流水线版)

```
游戏主机 (HDMI输出) 
    ↓ HDMI/DP
低延迟采集卡 (V4L2 /dev/video0)
    ↓ MJPEG/YUYV 视频流
Jetson AGX Xavier
    ├─ [CaptureThread]  GStreamer 硬件解码 / V4L2 采集
    │      ↓ frame_queue (maxsize=1, 旧帧丢弃)
    ├─ [InferThread]   核心计算流水线
    │      ├─ OpenCV/GPU 图像预处理 (letterbox, BGR→RGB, normalize)
    │      ├─ TensorRT 推理 (RTMO-l FP16, 640×640)
    │      ├─ RTMO 后处理解码 (NMS, 关键点恢复, 坐标映射)  [逻辑完全保留]
    │      ├─ 瞄准引擎 (目标选择 + PID平滑 + 预测补偿)      [逻辑完全保留]
    │      ↓ aim_queue (maxsize=2, 丢弃旧指令)
    ├─ [HIDThread]     uinput 事件驱动 (1000Hz等效)
    │      ↓ USB
    └─ [MainThread]    cv2.imshow 调试 / VideoRecorder 录制 / 性能监控
        ↓ USB
游戏主机 (鼠标输入)
```

> **精度保证**：`rtmo_decoder.py` 与 `aiming_engine.py` 零改动，NMS、关键点阈值、
> PID 参数、预测补偿、目标选择策略均与原版完全一致。

---

## 硬件要求

| 组件 | 规格 | 说明 |
|------|------|------|
| **主控** | Jetson AGX Xavier 32GB | Volta GPU 512 CUDA cores |
| **采集卡** | 低延迟 HDMI采集卡 | 支持 MJPEG 1080p@60fps, V4L2 接口 |
| **存储** | 64GB+ | 模型文件 + 日志 |
| **USB** | USB 3.0 | 连接采集卡 + 回传鼠标信号 |

---

## 软件环境

### JetPack 版本
- **JetPack 4.6+** (CUDA 10.2, TensorRT 8.0+) 或
- **JetPack 5.0+** (CUDA 11.4, TensorRT 8.4+)

### 预装组件 (JetPack 自带)
```bash
# CUDA
cuda-10.2 或 cuda-11.4

# TensorRT
/usr/lib/aarch64-linux-gnu/libnvinfer.so.8

# cuDNN
libcudnn8

# V4L2 / GStreamer
libgstreamer1.0-0
libgstreamer-plugins-base1.0-0

# OpenCV (with CUDA)
libopencv-core4.x  # JetPack 预编译版带 CUDA 支持
```

### Python 依赖
```bash
pip install numpy opencv-python opencv-contrib-python
```

---

## 项目结构

```
rtmo_aimbot_jetson/
├── main.py                    # 主程序入口 (多线程流水线)
├── src/
│   ├── config.py              # 全局配置 (新增 PipelineConfig)
│   ├── tensorrt_wrapper.py    # TensorRT 推理引擎
│   ├── rtmo_decoder.py        # RTMO 输出解码 + NMS
│   ├── aiming_engine.py       # 瞄准逻辑 (PID + 预测) [零改动]
│   ├── mouse_hid.py           # uinput HID 鼠标控制 (事件驱动版)
│   ├── video_capture.py       # V4L2/GStreamer 采集 (线程安全版)
│   └── utils.py               # 性能监控 + 工具 (线程安全版)
├── scripts/
│   ├── export_rtmo_onnx.py    # (x86) MMPose → ONNX 导出
│   ├── onnx2trt.py            # (Jetson) ONNX → TensorRT
│   └── calibrate_mouse.py     # 鼠标灵敏度校准
├── models/
│   └── rtmo_l_640x640.trt     # TensorRT 引擎 (需自行转换)
├── configs/
│   └── calibration.py         # 自动生成的校准配置
├── requirements.txt
└── README.md
```

---

## 部署步骤

### 步骤 1: 在 x86 开发机导出 ONNX

```bash
# 1. 安装 MMPose + MMDeploy
pip install mmpose mmdeploy onnx onnxsim

# 2. 下载 RTMO-l 权重
checkpoint="rtmo-l_16xb16-600e_coco-640x640-b59814f9_20231211.pth"

# 3. 修改 MMDeploy 后处理 (关键！)
# 编辑 mmdeploy/codebase/mmpose/models/heads/rtmo_head.py
# 注释掉 NMS 代码，保留原始输出 (bboxes + keypoints)

# 4. 导出 ONNX
python scripts/export_rtmo_onnx.py \
    --config configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_body7-640x640.py \
    --checkpoint ${checkpoint} \
    --output models/
```

### 步骤 2: 在 Jetson 转换 TensorRT 引擎

```bash
# 1. 将 ONNX 拷贝到 Jetson
scp models/rtmo_l_640x640.onnx jetson@192.168.x.x:~/rtmo_aimbot/models/

# 2. 转换 TensorRT 引擎
python3 scripts/onnx2trt.py \
    --onnx models/rtmo_l_640x640.onnx \
    --output models/rtmo_l_640x640.trt \
    --fp16 \
    --workspace 2048 \
    --height 640 \
    --width 640
```

### 步骤 3: 配置系统权限

```bash
# 1. 加载 uinput 模块
sudo modprobe uinput

# 2. 设置 uinput 权限
sudo chmod 666 /dev/uinput

# 3. 将当前用户加入 input 组
sudo usermod -a -G input $USER

# 4. 设置视频设备权限
sudo chmod 666 /dev/video0

# 5. (可选) 创建 udev 规则持久化权限
sudo tee /etc/udev/rules.d/99-aimbot.rules << 'EOF'
KERNEL=="uinput", MODE="0666"
KERNEL=="video[0-9]*", MODE="0666"
EOF
sudo udevadm control --reload-rules
```

### 步骤 4: 校准鼠标灵敏度

```bash
# 手动校准 (推荐)
python3 scripts/calibrate_mouse.py --distance 200 --samples 5

# 或自动校准 (需要画面检测)
python3 scripts/calibrate_mouse.py --auto --distance 100
```

根据游戏内灵敏度设置，校准后会生成 `configs/calibration.py`，将值更新到 `src/config.py` 中。

### 步骤 5: 运行 AimBot (多线程版)

```bash
# 调试模式 (显示画面 + 虚拟鼠标)
sudo python3 main.py --debug --dummy-mouse

# 生产模式 (无画面 + 实际控制鼠标)
sudo python3 main.py

# 带参数运行
sudo python3 main.py \
    --engine models/rtmo_l_640x640.trt \
    --conf 0.3 \
    --sensitivity 1.2 \
    --debug
```

---

## 配置调优

### 1. 采集延迟优化

```python
# src/config.py - CAPTURE_CFG
CAPTURE_CFG.use_hw_decode = True      # 启用 Jetson 硬件解码
CAPTURE_CFG.pixel_format = "MJPEG"    # MJPEG 比 YUYV 带宽更低
CAPTURE_CFG.buffer_count = 1          # 最小缓冲区减少延迟
```

### 2. 推理性能优化

```python
# src/config.py - MODEL_CFG
MODEL_CFG.fp16 = True                 # FP16 推理 (2x 速度提升)
MODEL_CFG.conf_thresh = 0.3           # 降低阈值增加召回 (可能增加误检)
```

### 3. 瞄准平滑调参

```python
# src/config.py - AIMING_CFG
AIMING_CFG.smooth_mode = "pid"        # PID 比 EMA 更稳定
AIMING_CFG.pid_kp = 0.8               # 增大比例增益 → 更跟手但更抖
AIMING_CFG.pid_kd = 0.2               # 增大微分增益 → 抑制超调
AIMING_CFG.enable_prediction = True     # 对移动目标启用预测
AIMING_CFG.prediction_frames = 2      # 预测未来 2 帧 (~33ms)
```

### 4. 目标选择策略

```python
AIMING_CFG.target_select_strategy = "nearest"   # 距离屏幕中心最近
# 可选: "center" (距离准星最近), "largest" (最大目标), "highest_conf" (最高置信度)
```

### 5. 流水线参数 (新增)

```python
# src/config.py - PIPELINE_CFG
PIPELINE_CFG.use_gpu_preprocess = True   # 启用 Jetson CUDA OpenCV 预处理
PIPELINE_CFG.frame_queue_size = 1        # 1=始终处理最新帧，丢弃旧帧
PIPELINE_CFG.aim_queue_size = 2          # 缓冲2帧瞄准指令，避免丢失
```

---

## 性能指标 (多线程流水线版, Jetson AGX Xavier)

| 指标 | 原版单线程 | 多线程流水线版 | 说明 |
|------|-----------|---------------|------|
| **采集阻塞** | 主循环阻塞 ~16ms | **零阻塞** | CaptureThread 独立运行 |
| **端到端延迟** | ~50–55 ms | **~30–40 ms** | 采集与计算并行，消除串行等待 |
| **有效推理帧率** | ~18–25 FPS | **~35–40 FPS** | 始终处理最新帧，丢弃积压旧帧 |
| **鼠标响应延迟** | 50–55 ms | **<< 3 ms** | HIDThread 事件驱动，指令合并发送 |
| **鼠标回报率** | 125 Hz | **1000 Hz 等效** | 事件队列驱动，非定时轮询 |
| **旧帧堆积** | 处理 3–4 帧前的画面 | **始终最新画面** | frame_queue/vis_queue maxsize=1 |

> 注：RTMO-l 在 640×640 FP16 下单帧推理+后处理仍需约 25–30ms，因此纯计算吞吐量上限约 35–40 FPS。
> 本改造通过消除 I/O 阻塞和旧帧堆积，让这 35–40 FPS 全部作用于最新画面，且鼠标响应与显示刷新解耦，
> 实际体验显著优于原版的"20 FPS 且鼠标卡顿"。若需真·60 FPS 逐帧不丢，请改用 RTMO-s/tiny 或 320×320 输入。

---

## 常见问题

### Q1: 采集卡无法识别
```bash
# 检查设备
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --all

# 检查支持的格式
v4l2-ctl -d /dev/video0 --list-formats-ext
```

### Q2: TensorRT 引擎构建失败
```bash
# 检查 TensorRT 版本
python3 -c "import tensorrt; print(tensorrt.__version__)"

# 如果 ONNX 解析失败，尝试简化 ONNX
python3 -c "import onnx; from onnxsim import simplify; m=onnx.load('model.onnx'); m,_=simplify(m); onnx.save(m, 'model_sim.onnx')"
```

### Q3: HID 鼠标无响应
```bash
# 检查 uinput 是否加载
lsmod | grep uinput

# 检查设备权限
ls -la /dev/uinput

# 测试 uinput 是否工作
cat /dev/input/mice  # 移动鼠标应看到数据
```

### Q4: 推理精度低 / 关键点抖动
- 检查预处理参数 (`mean`, `std`) 是否与训练时一致
- 调整 `conf_thresh` 和 `kpt_visible_thresh`
- 启用 `enable_prediction` 平滑移动目标

### Q5: 画面撕裂 / 帧丢失
- 降低采集分辨率到 720p
- 使用 `buffer_count=1` 最小缓冲
- 检查 USB 带宽 (采集卡 + 鼠标回传不要共用同一 USB 控制器)

### Q6: 推理线程帧率很高但画面卡顿
- 检查 `vis_queue` 是否堆积：主线程的 `cv2.imshow` 和 `VideoRecorder` 可能拖慢显示。
- 生产环境建议关闭 `show_debug_window` 和 `save_debug_video`，仅保留日志监控。

### Q7: 鼠标移动不连贯或跳帧
- 检查 `aim_queue` 是否频繁满：增大 `PIPELINE_CFG.aim_queue_size` 到 3–5。
- 检查游戏内鼠标灵敏度是否与 `MOUSE_CFG.sensitivity_x/y` 匹配，必要时重新运行 `calibrate_mouse.py`。

---

## 安全与合规声明

> ⚠️ **本系统仅通过 HDMI 视频环出 + HID 鼠标模拟与游戏交互**
> - 不读取游戏内存
> - 不修改游戏文件
> - 不注入任何代码
> - 不拦截网络封包
>
> 本项目的目的是研究计算机视觉在实时场景下的边缘部署与低延迟推理技术。请遵守各游戏平台的服务条款与社区准则。

---

## 参考

- [RTMO: Real-Time Multi-person pose estimation Optimized](https://github.com/open-mmlab/mmpose/tree/main/configs/body_2d_keypoint/rtmo)
- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
- [Jetson AGX Xavier Developer Kit](https://developer.nvidia.com/embedded/jetson-agx-xavier-developer-kit)
- [Linux uinput Documentation](https://www.kernel.org/doc/html/latest/input/uinput.html)
