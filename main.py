#!/usr/bin/env python3
"""
RTMO AimBot 多线程流水线主程序
部署于 Jetson AGX Xavier 32GB

支持后端切换: TensorRT (.trt) | ONNX Runtime (.onnx)

流水线架构：
  [CaptureThread]  ──frame_queue(maxsize=1,丢弃旧帧)──>  [InferThread]
    (V4L2/GStreamer)                                        (预处理→推理→解码→瞄准)
                                                                  │
  [HIDThread]    <──aim_queue(maxsize=2,丢弃旧指令)──────┘
    (uinput 事件驱动)                                          │
                                                                  │
  [MainThread]   <──vis_queue(maxsize=1,丢弃旧帧)──────┘
    (cv2.imshow调试显示 / 视频录制 / 性能监控)
"""
import os
import sys
import time
import logging
import argparse
import threading
import queue
from typing import Optional, Tuple

import numpy as np
import cv2

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODEL_CFG, CAPTURE_CFG, AIMING_CFG, MOUSE_CFG, SYS_CFG, PIPELINE_CFG
)
from src.rtmo_decoder import RTMODecoder, draw_debug_info
from src.aiming_engine import AimingEngine
from src.mouse_hid import create_mouse_controller
from src.video_capture import create_capture
from src.utils import (
    setup_logging, is_running, PerformanceMonitor,
    PerformanceMetrics, VideoRecorder
)

logger = logging.getLogger(__name__)

# ============================================================================
# 根据配置动态选择推理后端
# ============================================================================
if MODEL_CFG.backend == "tensorrt":
    from src.tensorrt_wrapper import TrtInferenceEngine, preprocess_image, preprocess_image_fast
    EngineClass = TrtInferenceEngine
    logger.info(f"使用后端: TensorRT ({MODEL_CFG.engine_path})")
elif MODEL_CFG.backend == "onnxruntime":
    from src.onnxruntime_wrapper import OnnxRuntimeEngine, preprocess_image, preprocess_image_fast
    EngineClass = OnnxRuntimeEngine
    logger.info(f"使用后端: ONNX Runtime ({MODEL_CFG.engine_path})")
else:
    raise ValueError(
        f"不支持的 backend: {MODEL_CFG.backend}\\n"
        f"可选值: 'tensorrt' | 'onnxruntime'"
    )


class ThreadSafeState:
    """线程共享状态 (鼠标位置 + 锁定状态)"""

    def __init__(self):
        self.mouse_x = CAPTURE_CFG.capture_width / 2
        self.mouse_y = CAPTURE_CFG.capture_height / 2
        self.lock = threading.Lock()
        self.target_locked = False

    def update_mouse(self, dx: float, dy: float):
        with self.lock:
            self.mouse_x += dx
            self.mouse_y += dy

    def get_mouse(self) -> Tuple[float, float]:
        with self.lock:
            return self.mouse_x, self.mouse_y

    def set_target_locked(self, locked: bool):
        with self.lock:
            self.target_locked = locked


class AimBotPipeline:
    """多线程流水线控制器"""

    def __init__(self, engine_path: str, dummy_mouse: bool = False):
        logger.info("=" * 60)
        logger.info("RTMO AimBot 多线程流水线初始化")
        logger.info(f"推理后端: {MODEL_CFG.backend}")
        logger.info("=" * 60)

        self.engine_path = engine_path
        self.dummy_mouse = dummy_mouse
        self.state = ThreadSafeState()

        # 线程控制
        self._running = True
        self._threads = []

        # 队列 (maxsize=1 实现最新数据丢弃旧数据，保证低延迟)
        self.frame_queue = queue.Queue(maxsize=PIPELINE_CFG.frame_queue_size)
        self.aim_queue = queue.Queue(maxsize=PIPELINE_CFG.aim_queue_size)
        self.vis_queue = queue.Queue(maxsize=PIPELINE_CFG.vis_queue_size)

        # 1. 加载推理引擎 (根据配置自动选择 TensorRT / ONNX Runtime)
        logger.info(f"加载模型: {engine_path}")
        self.engine = EngineClass(engine_path)

        # 2. 初始化视频采集
        logger.info(f"初始化视频采集: {CAPTURE_CFG.device}")
        self.capture = create_capture(use_gstreamer=CAPTURE_CFG.use_hw_decode)

        # 3. 创建鼠标控制器
        logger.info("创建鼠标控制器")
        self.mouse = create_mouse_controller(dummy=dummy_mouse)

        # 4. 初始化瞄准引擎 (单实例，仅 InferThread 访问，无线程竞争)
        self.aiming = AimingEngine(
            screen_width=CAPTURE_CFG.capture_width,
            screen_height=CAPTURE_CFG.capture_height
        )

        # 5. 性能监控
        self.monitor = PerformanceMonitor(SYS_CFG.perf_window_size)

        # 6. 调试视频录制
        self.recorder = None
        if SYS_CFG.save_debug_video:
            self.recorder = VideoRecorder(
                SYS_CFG.debug_video_path,
                fps=30,
                resolution=(CAPTURE_CFG.capture_width, CAPTURE_CFG.capture_height)
            )
            self.recorder.start()

        # 预热推理
        self._warmup()

    def _warmup(self):
        logger.info("预热推理引擎...")
        dummy_input = np.zeros((1, 3, MODEL_CFG.input_height, MODEL_CFG.input_width),
                               dtype=np.float32)
        for _ in range(3):
            self.engine.infer(dummy_input)
        logger.info("预热完成")

    def start(self):
        """启动所有工作线程"""
        logger.info("启动工作线程...")

        t_capture = threading.Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        t_infer = threading.Thread(target=self._infer_loop, name="InferThread", daemon=True)
        t_hid = threading.Thread(target=self._hid_loop, name="HIDThread", daemon=True)

        self._threads = [t_capture, t_infer, t_hid]
        for t in self._threads:
            t.start()
            logger.info(f"  {t.name} 已启动")

    def _capture_loop(self):
        """采集线程：持续读取，最新帧入队（满则丢弃旧帧）"""
        logger.info("采集线程启动")
        drop_count = 0

        while is_running() and self._running:
            ret, frame = self.capture.read()

            if not ret or frame is None:
                time.sleep(0.001)
                continue

            # 非阻塞入队，满则丢弃旧帧（降低延迟）
            try:
                self.frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    _ = self.frame_queue.get_nowait()
                    drop_count += 1
                    self.frame_queue.put_nowait(frame)
                except queue.Empty:
                    pass

            if drop_count > 0 and drop_count % 300 == 0:
                logger.info(f"采集线程已丢弃 {drop_count} 帧旧帧（降低延迟）")
                drop_count = 0

    def _infer_loop(self):
        """推理线程：预处理 → 推理 → 解码 → 瞄准"""
        logger.info("推理线程启动")

        # 本地 decoder 实例，确保参数与 MODEL_CFG 实时一致
        decoder = RTMODecoder(
            conf_thresh=MODEL_CFG.conf_thresh,
            nms_thresh=MODEL_CFG.nms_thresh,
            max_detections=MODEL_CFG.max_detections,
            num_keypoints=MODEL_CFG.num_keypoints
        )

        while is_running() and self._running:
            try:
                frame = self.frame_queue.get(timeout=PIPELINE_CFG.queue_timeout)
            except queue.Empty:
                continue

            orig_h, orig_w = frame.shape[:2]

            # 1. 预处理 (GPU优先，失败自动回退CPU，保持参数一致)
            t0 = time.time()
            try:
                if (PIPELINE_CFG.use_gpu_preprocess and
                        hasattr(cv2, 'cuda') and
                        cv2.cuda.getCudaEnabledDeviceCount() > 0):
                    preprocessed, scale, pad_offset = preprocess_image_fast(
                        frame,
                        target_size=(MODEL_CFG.input_width, MODEL_CFG.input_height)
                    )
                else:
                    preprocessed, scale, pad_offset = preprocess_image(
                        frame,
                        target_size=(MODEL_CFG.input_width, MODEL_CFG.input_height),
                        mean=MODEL_CFG.mean,
                        std=MODEL_CFG.std
                    )
            except Exception as e:
                logger.warning(f"预处理异常，回退CPU: {e}")
                preprocessed, scale, pad_offset = preprocess_image(
                    frame,
                    target_size=(MODEL_CFG.input_width, MODEL_CFG.input_height),
                    mean=MODEL_CFG.mean,
                    std=MODEL_CFG.std
                )
            t_preprocess = (time.time() - t0) * 1000

            # 2. 推理 (TensorRT 或 ONNX Runtime)
            t0 = time.time()
            outputs = self.engine.infer(preprocessed)
            t_infer = (time.time() - t0) * 1000

            # 3. 后处理解码 (保持原有精度逻辑不变)
            t0 = time.time()
            persons = decoder.decode(
                outputs, scale, pad_offset, (orig_h, orig_w)
            )
            t_postprocess = (time.time() - t0) * 1000

            # 4. 瞄准逻辑
            t0 = time.time()
            mouse_x, mouse_y = self.state.get_mouse()
            result = self.aiming.process(persons, mouse_x, mouse_y)
            t_aiming = (time.time() - t0) * 1000

            # 5. 组装结果
            total_latency = t_preprocess + t_infer + t_postprocess + t_aiming

            target_person = None
            aim_point = None
            dx, dy = 0.0, 0.0
            should_fire = False

            if result is not None:
                dx, dy, target_person = result
                if target_person is not None:
                    aim_point = decoder.get_aim_point(target_person)[:2]

                # 自动开火判断 (保持原有逻辑)
                if MOUSE_CFG.auto_fire and abs(dx) < 20 and abs(dy) < 20:
                    should_fire = True

                # 发送到 HID 队列
                try:
                    self.aim_queue.put_nowait((dx, dy, should_fire))
                except queue.Full:
                    try:
                        _ = self.aim_queue.get_nowait()
                        self.aim_queue.put_nowait((dx, dy, should_fire))
                    except queue.Empty:
                        pass

            self.state.set_target_locked(target_person is not None)

            # 6. 可视化数据入队
            if SYS_CFG.show_debug_window or self.recorder is not None:
                vis_data = {
                    'frame': frame,
                    'persons': persons,
                    'target_person': target_person,
                    'aim_point': aim_point,
                    'latency_ms': total_latency,
                }
                try:
                    self.vis_queue.put_nowait(vis_data)
                except queue.Full:
                    try:
                        _ = self.vis_queue.get_nowait()
                        self.vis_queue.put_nowait(vis_data)
                    except queue.Empty:
                        pass

            # 性能统计
            metrics = PerformanceMetrics(
                capture_fps=self.capture.get_fps(),
                inference_ms=t_infer,
                preprocess_ms=t_preprocess,
                postprocess_ms=t_postprocess,
                total_latency_ms=total_latency,
                mouse_move_ms=0.0,
                persons_detected=len(persons),
                target_locked=target_person is not None
            )
            self.monitor.record(metrics)

    def _hid_loop(self):
        """HID线程：事件驱动，高频率发送鼠标事件"""
        logger.info("HID线程启动")

        while is_running() and self._running:
            try:
                dx, dy, should_fire = self.aim_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            # 合并同批所有后续瞄准指令（减少系统调用）
            total_dx, total_dy = dx, dy
            fire_flags = [should_fire]
            while not self.aim_queue.empty():
                try:
                    dx2, dy2, f2 = self.aim_queue.get_nowait()
                    total_dx += dx2
                    total_dy += dy2
                    fire_flags.append(f2)
                except queue.Empty:
                    break

            # 发送合并后的移动
            self.mouse.move(total_dx, total_dy)

            # 更新已发送的鼠标位置（供 InferThread 下一帧使用）
            self.state.update_mouse(total_dx, total_dy)

            # 自动开火（任一帧要求开火则开火）
            if any(fire_flags):
                self.mouse.click("left")

    def run(self):
        """主线程：调试显示 + 视频录制 + 性能监控"""
        logger.info("主循环启动（可视化线程）")

        # 等待推理线程产生第一帧可视化数据
        wait_start = time.time()
        while self.vis_queue.empty() and (time.time() - wait_start) < 10.0:
            time.sleep(0.01)

        if self.vis_queue.empty():
            logger.warning("可视化队列无数据，推理线程可能未成功启动")

        frame_interval = 1.0 / SYS_CFG.target_fps
        last_frame_time = time.time()

        while is_running() and self._running:
            # 从可视化队列取数据（非阻塞，丢弃旧帧）
            try:
                vis_data = self.vis_queue.get(timeout=0.001)
            except queue.Empty:
                time.sleep(0.001)
                continue

            frame = vis_data['frame']
            persons = vis_data['persons']
            target_person = vis_data['target_person']
            aim_point = vis_data['aim_point']
            total_latency = vis_data['latency_ms']

            # 调试显示 (OpenCV必须在主线程)
            if SYS_CFG.show_debug_window:
                vis_frame = draw_debug_info(
                    frame, persons, target_person, aim_point, total_latency
                )

                # 添加性能信息
                perf_text = self.monitor.get_summary()
                cv2.putText(vis_frame, perf_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 缩放显示
                if SYS_CFG.debug_scale != 1.0:
                    dw = int(vis_frame.shape[1] * SYS_CFG.debug_scale)
                    dh = int(vis_frame.shape[0] * SYS_CFG.debug_scale)
                    vis_frame = cv2.resize(vis_frame, (dw, dh))

                cv2.imshow("RTMO AimBot Debug", vis_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self._running = False
                    break

            # 录制调试视频
            if self.recorder is not None:
                vis_frame = draw_debug_info(
                    frame, persons, target_person, aim_point, total_latency
                )
                self.recorder.write(vis_frame)

            # 显示帧率控制（仅影响可视化，不影响核心流水线）
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

        self.shutdown()

    def shutdown(self):
        """关闭所有资源"""
        logger.info("正在关闭流水线...")
        self._running = False

        # 等待线程结束
        for t in self._threads:
            logger.info(f"等待 {t.name} 结束...")
            t.join(timeout=2.0)
            if t.is_alive():
                logger.warning(f"{t.name} 未在2秒内结束")

        self.monitor.print_stats()

        if self.recorder is not None:
            self.recorder.stop()

        self.mouse.close()
        self.capture.release()
        self.engine.release()
        cv2.destroyAllWindows()

        logger.info("已安全退出")


def main():
    parser = argparse.ArgumentParser(description="RTMO AimBot for Jetson AGX Xavier (Pipeline Edition)")
    parser.add_argument("--engine", type=str, default=MODEL_CFG.engine_path,
                       help="模型路径 (.trt 或 .onnx)")
    parser.add_argument("--backend", type=str, default=MODEL_CFG.backend,
                       help="推理后端: tensorrt | onnxruntime")
    parser.add_argument("--dummy-mouse", action="store_true",
                       help="使用虚拟鼠标模式 (不实际控制鼠标)")
    parser.add_argument("--debug", action="store_true",
                       help="显示调试窗口")
    parser.add_argument("--record", action="store_true",
                       help="录制调试视频")
    parser.add_argument("--device", type=str, default=CAPTURE_CFG.device,
                       help="视频设备路径")
    parser.add_argument("--conf", type=float, default=MODEL_CFG.conf_thresh,
                       help="检测置信度阈值")
    parser.add_argument("--sensitivity", type=float, default=MOUSE_CFG.sensitivity_x,
                       help="鼠标灵敏度")

    args = parser.parse_args()

    # 应用命令行参数
    SYS_CFG.show_debug_window = args.debug or SYS_CFG.show_debug_window
    SYS_CFG.save_debug_video = args.record or SYS_CFG.save_debug_video
    CAPTURE_CFG.device = args.device
    MODEL_CFG.conf_thresh = args.conf
    MOUSE_CFG.sensitivity_x = args.sensitivity
    MOUSE_CFG.sensitivity_y = args.sensitivity

    # 命令行可覆盖后端和模型路径
    if args.backend:
        MODEL_CFG.backend = args.backend
    if args.engine:
        MODEL_CFG.engine_path = args.engine

    # 设置日志
    setup_logging(SYS_CFG.log_level)

    # 检查权限
    if os.geteuid() != 0 and not args.dummy_mouse:
        logger.warning("未以 root 运行，HID 设备可能需要额外权限")
        logger.warning("建议: sudo python3 main.py")

    # 启动
    pipeline = AimBotPipeline(MODEL_CFG.engine_path, dummy_mouse=args.dummy_mouse)
    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
