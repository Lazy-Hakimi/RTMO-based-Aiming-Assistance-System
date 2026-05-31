"""
RTMO AimBot 全局配置
适配 Jetson AGX Xavier 32GB
"""
import os
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置"""
    # TensorRT 引擎路径
    engine_path: str = "models/rtmo_l_640x640.trt"
    onnx_path: str = "models/rtmo_l_640x640.onnx"

    # 输入尺寸 (RTMO-l 默认 640x640)
    input_width: int = 640
    input_height: int = 640

    # 预处理参数 (MMPose 标准归一化)
    mean: Tuple[float, float, float] = (123.675, 116.28, 103.53)
    std: Tuple[float, float, float] = (58.395, 57.12, 57.375)

    # 置信度阈值
    conf_thresh: float = 0.3

    # NMS 参数
    nms_thresh: float = 0.65
    max_detections: int = 10

    # 关键点数量 (COCO: 17点)
    num_keypoints: int = 17

    # 使用 FP16 推理
    fp16: bool = True

    # 工作空间大小 (MB)
    workspace_mb: int = 2048


@dataclass
class CaptureConfig:
    """视频采集配置"""
    # V4L2 设备路径 (HDMI采集卡)
    device: str = "/dev/video0"

    # 采集分辨率 (采集卡输出，不一定是模型输入)
    capture_width: int = 1920
    capture_height: int = 1080
    capture_fps: int = 60

    # 像素格式 (MJPEG 或 YUYV)
    pixel_format: str = "MJPEG"

    # 缓冲区数量
    buffer_count: int = 4

    # 是否使用 GPU 加速解码 (Jetson 硬件解码)
    use_hw_decode: bool = True

    # 采集超时 (ms)
    capture_timeout_ms: int = 1000


@dataclass
class AimingConfig:
    """瞄准逻辑配置"""
    # 目标关键点选择
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
    # 5: left_shoulder, 6: right_shoulder
    # 优先瞄准头部，如果头部不可见则 fallback 到躯干
    priority_keypoints: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    fallback_keypoints: List[int] = field(default_factory=lambda: [5, 6, 11, 12])

    # 关键点可见性阈值
    kpt_visible_thresh: float = 0.3

    # 目标选择策略: "nearest", "center", "largest", "highest_conf"
    target_select_strategy: str = "nearest"

    # 瞄准区域 (屏幕中心区域，相对坐标 0-1)
    # 只锁定进入该区域的目标
    aim_region_x: Tuple[float, float] = (0.3, 0.7)  # 水平范围
    aim_region_y: Tuple[float, float] = (0.2, 0.8)  # 垂直范围

    # 平滑系数 (EMA: 0.0-1.0, 越大越跟手但越抖)
    # 或使用 PID 控制
    smooth_mode: str = "pid"  # "ema" 或 "pid"

    # EMA 配置
    ema_alpha: float = 0.35

    # PID 配置
    pid_kp: float = 0.8   # 比例增益
    pid_ki: float = 0.05  # 积分增益
    pid_kd: float = 0.2   # 微分增益
    pid_integral_limit: float = 50.0  # 积分限幅

    # 鼠标移动速度限制 (像素/帧)
    max_move_per_frame: float = 80.0

    # 最小移动阈值 (避免微抖动)
    min_move_threshold: float = 2.0

    # 预测补偿 (根据目标移动速度预测下一帧位置)
    enable_prediction: bool = True
    prediction_frames: int = 2  # 预测未来帧数

    # 人体比例过滤 (排除过大/过小的误检)
    min_person_height_ratio: float = 0.15  # 相对画面高度最小比例
    max_person_height_ratio: float = 0.9     # 相对画面高度最大比例


@dataclass
class MouseConfig:
    """鼠标控制配置"""
    # HID 设备路径
    uinput_device: str = "/dev/uinput"

    # 鼠标 DPI 映射系数 (将像素偏移转换为鼠标单位)
    # 需要根据实际游戏灵敏度校准
    sensitivity_x: float = 1.0
    sensitivity_y: float = 1.0

    # 鼠标事件发送间隔 (秒)
    # 多线程版改为 1ms (1000Hz 等效)
    send_interval: float = 0.001

    # 是否启用鼠标控制 (调试时可关闭)
    enabled: bool = True

    # 鼠标按键模拟 (如需自动开火)
    auto_fire: bool = False
    auto_fire_delay: float = 0.05  # 开火间隔


@dataclass
class SystemConfig:
    """系统配置"""
    # 推理线程数
    infer_threads: int = 1

    # 是否显示调试画面
    show_debug_window: bool = True

    # 调试窗口缩放
    debug_scale: float = 0.5

    # 性能统计窗口大小
    perf_window_size: int = 30

    # 日志级别
    log_level: str = "INFO"

    # 保存调试视频
    save_debug_video: bool = False
    debug_video_path: str = "output/debug.mp4"

    # 主循环目标帧率
    target_fps: int = 60

    # 使用 CUDA Graph 加速 (TensorRT 8.6+)
    use_cuda_graph: bool = False

    # 零拷贝内存 (Jetson 共享内存架构优势)
    use_zero_copy: bool = True


@dataclass
class PipelineConfig:
    """多线程流水线配置"""
    # 队列大小 (1=最新帧/指令丢弃旧数据，保证低延迟)
    frame_queue_size: int = 1
    aim_queue_size: int = 2   # 稍大以避免指令丢失导致鼠标状态漂移
    vis_queue_size: int = 1

    # 队列阻塞超时 (秒)
    queue_timeout: float = 0.5

    # 是否使用 GPU 预处理 (Jetson CUDA OpenCV)
    use_gpu_preprocess: bool = True

    # HID 发送频率 (Hz), 0表示由mouse_hid内部事件驱动
    hid_send_hz: int = 1000

    # 推理线程数 (Jetson CPU弱，保持1)
    infer_threads: int = 1

    # 是否启用采集-推理双缓冲重叠 (高级，需TensorRT上下文支持)
    enable_overlap: bool = False

    # 可视化线程独立运行 (OpenCV imshow必须在主线程，故保持False)
    separate_vis_thread: bool = False


# 全局配置实例
MODEL_CFG = ModelConfig()
CAPTURE_CFG = CaptureConfig()
AIMING_CFG = AimingConfig()
MOUSE_CFG = MouseConfig()
SYS_CFG = SystemConfig()
PIPELINE_CFG = PipelineConfig()

# COCO 关键点名称映射 (用于调试)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # 头部
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 上肢
    (5, 11), (6, 12), (11, 12),  # 躯干
    (11, 13), (13, 15), (12, 14), (14, 16)  # 下肢
]
