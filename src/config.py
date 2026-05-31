"""
RTMO AimBot Enhanced 全局配置
适配 Jetson AGX Xavier 32GB + ESP32-S3 HID 鼠标设备

架构：
  Xavier (推理端) --UART/USB--> ESP32-S3 (HID设备端) --USB--> 游戏主机
  Xavier 负责: 视频采集、TensorRT推理、后处理解码、身体朝向估算、
               多目标优先级选择、压枪补偿、准星校准
  ESP32-S3 负责: 接收指令、模拟HID鼠标事件输出
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    engine_path: str = "models/rtmo_l_640x640.trt"
    onnx_path: str = "models/rtmo_l_640x640.onnx"
    input_width: int = 640
    input_height: int = 640
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)
    conf_thresh: float = 0.3
    nms_thresh: float = 0.65
    max_detections: int = 10
    num_keypoints: int = 17  # COCO格式17点
    fp16: bool = True
    workspace_mb: int = 2048


@dataclass
class CaptureConfig:
    """视频采集配置"""
    device: str = "/dev/video0"
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 60
    pixel_format: str = "MJPEG"
    buffer_count: int = 4
    use_hw_decode: bool = True
    capture_timeout_ms: int = 1000


@dataclass
class BodyOrientationConfig:
    """身体朝向估算配置"""
    enabled: bool = True
    # 关键点索引 (COCO格式)
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder
    # 7: left_elbow, 8: right_elbow
    # 9: left_wrist, 10: right_wrist
    # 11: left_hip, 12: right_hip
    # 13: left_knee, 14: right_knee
    # 15: left_ankle, 16: right_ankle

    # 用于朝向估算的关键点
    shoulder_left_idx: int = 5
    shoulder_right_idx: int = 6
    hip_left_idx: int = 11
    hip_right_idx: int = 12
    nose_idx: int = 0

    # 可见性阈值
    kpt_visible_thresh: float = 0.3

    # 朝向分类角度阈值 (度)
    # 肩线与水平面的夹角用于判断朝向
    facing_front_thresh: float = 30.0   # 小于此角度认为正面朝向
    facing_side_thresh: float = 60.0    # 小于此角度认为侧面，大于则背面

    # 肩宽比阈值 (肩宽/髋宽)
    shoulder_hip_ratio_front: float = 1.3   # 大于此值为正面
    shoulder_hip_ratio_back: float = 0.8    # 小于此值为背面

    # 历史平滑帧数
    history_smooth_frames: int = 3

    # 朝向权重 (用于优先级计算)
    facing_front_weight: float = 1.0   # 正面目标威胁最高
    facing_side_weight: float = 0.7    # 侧面
    facing_back_weight: float = 0.4    # 背面威胁最低


@dataclass
class TargetPriorityConfig:
    """多目标优先级选择配置"""
    # 策略模式
    strategy: str = "composite"  # "composite" (综合权重), "nearest", "threat", "orientation"

    # 权重系数 (用于composite模式)
    w_distance: float = 0.35     # 距离权重 (离准星越近分越高)
    w_threat: float = 0.30       # 威胁度权重 (正面朝向威胁更高)
    w_orientation: float = 0.20  # 朝向权重 (背面/侧面更容易击杀)
    w_size: float = 0.10         # 大小权重 (越大越明显)
    w_confidence: float = 0.05   # 置信度权重

    # 距离分数参数
    max_aim_distance: float = 500.0  # 最大瞄准距离(像素)
    distance_decay: str = "gaussian"  # "linear" 或 "gaussian"

    # 锁定保持参数
    lock_keep_frames: int = 5    # 丢失目标后保持锁定的帧数
    lock_iou_threshold: float = 0.3  # 目标切换的最小IoU阈值

    # 瞄准区域 (屏幕中心区域，相对坐标 0-1)
    aim_region_x: Tuple[float, float] = (0.3, 0.7)
    aim_region_y: Tuple[float, float] = (0.2, 0.8)

    # 是否启用朝向感知
    use_orientation: bool = True
    # 是否优先击杀背身目标
    prioritize_back: bool = True


@dataclass
class RecoilConfig:
    """压枪补偿配置"""
    enabled: bool = True

    # 压枪模式
    mode: str = "adaptive"  # "adaptive" (自适应), "pattern" (固定模式), "hybrid" (混合)

    # 固定模式参数 (像素/发)
    # 垂直补偿 (每发子弹的向上偏移补偿量)
    base_compensation_y: float = 3.0  # 基础垂直补偿
    base_compensation_x: float = 0.5  # 基础水平补偿 (随机左右漂移)

    # 武器配置 (不同武器的后坐力模式)
    # 格式: {weapon_name: {"vertical_per_shot": float, "horizontal_drift": float, "pattern": list}}
    weapon_profiles: dict = field(default_factory=lambda: {
        "default": {
            "vertical_per_shot": 2.5,   # 每发垂直偏移量
            "horizontal_drift": 0.3,    # 水平漂移范围
            "max_compensation": 25.0,   # 最大单帧补偿量
            "recovery_rate": 0.3,       # 枪口回降速度 (每帧)
            "bullets_per_pattern": 30,  # 一个模式的子弹数
        },
        "rifle": {
            "vertical_per_shot": 2.0,
            "horizontal_drift": 0.4,
            "max_compensation": 20.0,
            "recovery_rate": 0.35,
            "bullets_per_pattern": 30,
        },
        "smg": {
            "vertical_per_shot": 1.5,
            "horizontal_drift": 0.6,
            "max_compensation": 15.0,
            "recovery_rate": 0.4,
            "bullets_per_pattern": 40,
        },
        "lmg": {
            "vertical_per_shot": 3.0,
            "horizontal_drift": 0.5,
            "max_compensation": 30.0,
            "recovery_rate": 0.2,
            "bullets_per_pattern": 100,
        },
    })

    current_weapon: str = "rifle"  # 当前武器类型

    # 自适应压枪参数
    adaptive_window_size: int = 10  # 自适应窗口大小(发)
    adaptive_learning_rate: float = 0.1  # 自适应学习率

    # 开火检测参数
    fire_detect_threshold: int = 3  # 连续命中目标次数视为开火状态

    # 压枪模式限制
    max_shots_compensated: int = 30  # 最大连续压枪发数
    compensation_delay_frames: int = 0  # 开火后延迟帧数开始压枪


@dataclass
class CalibratorConfig:
    """准星校准配置"""
    enabled: bool = True

    # 校准模式
    mode: str = "manual"  # "manual" (手动), "auto" (自动), "semi_auto" (半自动)

    # 校准参数
    calibration_steps: int = 8  # 校准方向数 (8方向: 上下左右+对角)
    calibration_distance: int = 100  # 每次校准移动距离 (像素)

    # 自动校准参数
    auto_detect_target: bool = True  # 自动检测校准靶标
    target_template_size: int = 50   # 靶标模板大小

    # 校准结果文件
    calibration_file: str = "configs/calibration_matrix.json"

    # 灵敏度校准
    sensitivity_test_distances: List[int] = field(default_factory=lambda: [50, 100, 200])
    sensitivity_samples_per_distance: int = 3

    # 压枪校准
    recoil_calibration_shots: int = 10  # 压枪校准射击次数
    recoil_calibration_weapon: str = "rifle"


@dataclass
class ESP32BridgeConfig:
    """ESP32-S3 通信桥接配置"""
    # 通信模式
    mode: str = "serial"  # "serial" (UART), "udp" (WiFi UDP), "usb_cdc" (USB CDC)

    # UART配置 (mode="serial")
    serial_port: str = "/dev/ttyACM0"  # ESP32-S3连接到Xavier的串口
    serial_baudrate: int = 921600       # 波特率
    serial_timeout: float = 0.001       # 读超时(秒)

    # UDP配置 (mode="udp")
    udp_send_port: int = 8888      # Xavier发送端口
    udp_recv_port: int = 8889      # Xavier接收端口
    esp32_ip: str = "192.168.4.1"  # ESP32 Station模式IP (AP模式)

    # USB CDC配置 (mode="usb_cdc")
    usb_cdc_port: str = "/dev/ttyACM0"

    # 协议配置
    protocol_version: int = 1
    use_binary_protocol: bool = True  # True=二进制协议, False=文本协议(JSON)

    # 指令发送频率 (Hz)
    command_send_hz: int = 1000

    # 重连参数
    auto_reconnect: bool = True
    reconnect_interval: float = 1.0  # 重连间隔(秒)
    max_reconnect_attempts: int = 10

    # 心跳检测
    heartbeat_interval: float = 1.0  # 心跳间隔(秒)
    heartbeat_timeout: float = 3.0   # 心跳超时(秒)

    # 调试
    dummy_esp32: bool = False  # 虚拟ESP32模式 (本地测试)
    show_tx_rx: bool = False   # 显示发送/接收的原始数据


@dataclass
class AimingConfig:
    """瞄准逻辑配置 (保留原配置并扩展)"""
    # 目标关键点选择 (头部优先)
    priority_keypoints: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    fallback_keypoints: List[int] = field(default_factory=lambda: [5, 6, 11, 12])

    # 关键点可见性阈值
    kpt_visible_thresh: float = 0.3

    # 平滑模式
    smooth_mode: str = "pid"  # "ema" 或 "pid"

    # EMA配置
    ema_alpha: float = 0.35

    # PID配置
    pid_kp: float = 0.8
    pid_ki: float = 0.05
    pid_kd: float = 0.2
    pid_integral_limit: float = 50.0

    # 鼠标移动限制
    max_move_per_frame: float = 80.0
    min_move_threshold: float = 2.0

    # 预测补偿
    enable_prediction: bool = True
    prediction_frames: int = 2

    # 人体比例过滤
    min_person_height_ratio: float = 0.15
    max_person_height_ratio: float = 0.9

    # 新增: 是否启用身体朝向估算
    enable_orientation: bool = True

    # 新增: 是否启用压枪补偿
    enable_recoil_compensation: bool = True

    # 新增: 是否启用准星校准
    enable_calibration: bool = True

    # 瞄准偏移补偿 (用于校准后的微调)
    aim_offset_x: float = 0.0
    aim_offset_y: float = 0.0


@dataclass
class MouseConfig:
    """鼠标控制配置 (更新: 支持ESP32-S3和本地两种模式)"""
    # 鼠标模式
    mouse_mode: str = "esp32"  # "esp32" (通过ESP32-S3), "local" (本地uinput), "dummy" (虚拟)

    # 本地uinput配置 (mouse_mode="local")
    uinput_device: str = "/dev/uinput"

    # 灵敏度系数
    sensitivity_x: float = 1.0
    sensitivity_y: float = 1.0

    # 事件发送间隔
    send_interval: float = 0.001

    # 是否启用
    enabled: bool = True

    # 自动开火
    auto_fire: bool = False
    auto_fire_delay: float = 0.05

    # 新增: ESP32桥接配置引用
    esp32_auto_reconnect: bool = True


@dataclass
class SystemConfig:
    """系统配置"""
    infer_threads: int = 1
    show_debug_window: bool = True
    debug_scale: float = 0.5
    perf_window_size: int = 30
    log_level: str = "INFO"
    save_debug_video: bool = False
    debug_video_path: str = "output/debug.mp4"
    target_fps: int = 60
    use_cuda_graph: bool = False
    use_zero_copy: bool = True


@dataclass
class PipelineConfig:
    """多线程流水线配置"""
    frame_queue_size: int = 1
    aim_queue_size: int = 2
    vis_queue_size: int = 1
    queue_timeout: float = 0.5
    use_gpu_preprocess: bool = True
    hid_send_hz: int = 1000
    infer_threads: int = 1
    enable_overlap: bool = False
    separate_vis_thread: bool = False


# 全局配置实例
MODEL_CFG = ModelConfig()
CAPTURE_CFG = CaptureConfig()
AIMING_CFG = AimingConfig()
MOUSE_CFG = MouseConfig()
SYS_CFG = SystemConfig()
PIPELINE_CFG = PipelineConfig()

# 新增: 扩展配置实例
BODY_ORI_CFG = BodyOrientationConfig()
TARGET_PRIO_CFG = TargetPriorityConfig()
RECOIL_CFG = RecoilConfig()
CALIB_CFG = CalibratorConfig()
ESP32_CFG = ESP32BridgeConfig()

# COCO关键点名称映射
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# 身体朝向枚举
class BodyFacing:
    """身体朝向分类"""
    FRONT = "front"      # 正面朝向镜头
    LEFT_SIDE = "left"   # 左侧朝向镜头
    RIGHT_SIDE = "right" # 右侧朝向镜头
    BACK = "back"        # 背面朝向镜头
    UNKNOWN = "unknown"  # 未知

# 枪口朝向估算 (基于手臂关键点)
class MuzzleOrientation:
    """枪口朝向分类 (基于手臂姿态估算)"""
    FORWARD = "forward"    # 前向 (双手前伸)
    LEFT = "left"          # 左
    RIGHT = "right"        # 右
    UP = "up"              # 上
    DOWN = "down"          # 下
    UNKNOWN = "unknown"
