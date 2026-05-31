#!/usr/bin/env python3
"""
RTMO AimBot Enhanced - 增强版主程序
Jetson AGX Xavier (推理端) + ESP32-S3 (HID设备端)

新增功能:
1. 身体朝向估算与多目标优先级选择
2. 压枪补偿 (支持多种武器配置)
3. 准星校准系统 (自动/手动)
4. ESP32-S3 HID鼠标桥接

流水线架构:
  [CaptureThread]  --frame_queue-->  [InferThread]
    (V4L2采集)                        (预处理→推理→解码→瞄准→压枪)
                                            |
  [ESP32Bridge]  <----aim_queue----┘
    (UART→ESP32-S3)                       |
  [MainThread]   <----vis_queue----┘
    (显示+校准UI+性能监控)

按键控制:
  F1: 切换武器配置
  F2: 开关压枪补偿
  F3: 进入手动校准模式
  F4: 自动校准准星
  F5: 切换目标选择策略
  +/-: 调整灵敏度
  Q/E: 切换瞄准部位 (头/身)
  ESC: 退出校准模式
  q:   退出程序
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import (
    MODEL_CFG, CAPTURE_CFG, AIMING_CFG, MOUSE_CFG, SYS_CFG, PIPELINE_CFG,
    BODY_ORI_CFG, TARGET_PRIO_CFG, RECOIL_CFG, CALIB_CFG, ESP32_CFG,
    BodyFacing
)
from src.tensorrt_wrapper import TrtInferenceEngine, preprocess_image, preprocess_image_fast
from src.rtmo_decoder import RTMODecoder, draw_debug_info
from src.aiming_engine import AimingEngine
from src.mouse_hid import create_mouse_controller
from src.video_capture import create_capture
from src.utils import (
    setup_logging, is_running, PerformanceMonitor,
    PerformanceMetrics, VideoRecorder
)

logger = logging.getLogger(__name__)


class ThreadSafeState:
    """线程共享状态"""

    def __init__(self):
        self.mouse_x = CAPTURE_CFG.capture_width / 2
        self.mouse_y = CAPTURE_CFG.capture_height / 2
        self.lock = threading.Lock()
        self.target_locked = False
        self.should_fire = False

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

    def set_fire(self, fire: bool):
        with self.lock:
            self.should_fire = fire

    def get_fire(self) -> bool:
        with self.lock:
            return self.should_fire


class AimBotPipelineEnhanced:
    """增强版多线程流水线控制器"""

    def __init__(self, engine_path: str, dummy_mouse: bool = False, dummy_esp32: bool = False):
        logger.info("=" * 60)
        logger.info("RTMO AimBot Enhanced 多线程流水线初始化")
        logger.info("=" * 60)

        self.engine_path = engine_path
        self.dummy_mouse = dummy_mouse
        ESP32_CFG.dummy_esp32 = dummy_esp32
        self.state = ThreadSafeState()

        # 线程控制
        self._running = True
        self._threads = []

        # 队列
        self.frame_queue = queue.Queue(maxsize=PIPELINE_CFG.frame_queue_size)
        self.aim_queue = queue.Queue(maxsize=PIPELINE_CFG.aim_queue_size)
        self.vis_queue = queue.Queue(maxsize=PIPELINE_CFG.vis_queue_size)

        # 1. 加载TensorRT引擎
        logger.info(f"加载 TensorRT 引擎: {engine_path}")
        self.engine = TrtInferenceEngine(engine_path)

        # 2. 初始化视频采集
        logger.info(f"初始化视频采集: {CAPTURE_CFG.device}")
        self.capture = create_capture(use_gstreamer=CAPTURE_CFG.use_hw_decode)

        # 3. 创建鼠标控制器 (ESP32桥接模式)
        mouse_mode = "dummy" if dummy_mouse else MOUSE_CFG.mouse_mode
        logger.info(f"创建鼠标控制器: mode={mouse_mode}")
        self.mouse = create_mouse_controller(mode=mouse_mode)

        # 4. 初始化增强版瞄准引擎
        logger.info("初始化增强版瞄准引擎")
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

        # 7. 当前武器索引
        self.weapon_list = self.aiming.recoil_compensator.get_weapon_list()
        self.current_weapon_idx = 0

        # 预热
        self._warmup()

    def _warmup(self):
        logger.info("预热推理引擎...")
        dummy_input = np.zeros((1, 3, MODEL_CFG.input_height, MODEL_CFG.input_width),
                               dtype=np.float32)
        for _ in range(3):
            self.engine.infer(dummy_input)
        logger.info("预热完成")

    def start(self):
        logger.info("启动工作线程...")

        t_capture = threading.Thread(target=self._capture_loop, name="CaptureThread", daemon=True)
        t_infer = threading.Thread(target=self._infer_loop, name="InferThread", daemon=True)
        t_hid = threading.Thread(target=self._hid_loop, name="HIDThread", daemon=True)

        self._threads = [t_capture, t_infer, t_hid]
        for t in self._threads:
            t.start()
            logger.info(f"  {t.name} 已启动")

    def _capture_loop(self):
        """采集线程"""
        logger.info("采集线程启动")
        drop_count = 0

        while is_running() and self._running:
            ret, frame = self.capture.read()
            if not ret or frame is None:
                time.sleep(0.001)
                continue

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
                logger.info(f"采集线程已丢弃 {drop_count} 帧旧帧")
                drop_count = 0

    def _infer_loop(self):
        """推理线程 (增强版)"""
        logger.info("推理线程启动")

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

            # 1. 预处理
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

            # 2. TensorRT推理
            t0 = time.time()
            outputs = self.engine.infer(preprocessed)
            t_infer = (time.time() - t0) * 1000

            # 3. 后处理解码
            t0 = time.time()
            persons = decoder.decode(
                outputs, scale, pad_offset, (orig_h, orig_w)
            )
            t_postprocess = (time.time() - t0) * 1000

            # 4. 瞄准逻辑 (增强版)
            t0 = time.time()
            mouse_x, mouse_y = self.state.get_mouse()

            # 自动开火判断
            should_fire = False
            if MOUSE_CFG.auto_fire and persons:
                # 简单判断: 如果有目标且在准星附近
                should_fire = True

            self.state.set_fire(should_fire)

            result = self.aiming.process(persons, mouse_x, mouse_y, should_fire)
            t_aiming = (time.time() - t0) * 1000

            # 5. 组装结果
            total_latency = t_preprocess + t_infer + t_postprocess + t_aiming

            target_person = None
            aim_point = None
            dx, dy = 0.0, 0.0
            should_fire_out = False

            if result is not None:
                dx, dy, target_person = result
                if target_person is not None:
                    aim_point = decoder.get_aim_point(target_person)[:2]

                # 自动开火判断 (精确)
                if MOUSE_CFG.auto_fire and abs(dx) < 20 and abs(dy) < 20:
                    should_fire_out = True

                # 发送到HID队列
                try:
                    self.aim_queue.put_nowait((dx, dy, should_fire_out))
                except queue.Full:
                    try:
                        _ = self.aim_queue.get_nowait()
                        self.aim_queue.put_nowait((dx, dy, should_fire_out))
                    except queue.Empty:
                        pass

            self.state.set_target_locked(target_person is not None)

            # 6. 可视化数据入队
            if SYS_CFG.show_debug_window or self.recorder is not None:
                recoil_info = self.aiming.get_recoil_status()
                calib_info = self.aiming.get_calibration_info()

                vis_data = {
                    'frame': frame,
                    'persons': persons,
                    'target_person': target_person,
                    'aim_point': aim_point,
                    'latency_ms': total_latency,
                    'recoil_info': recoil_info,
                    'calib_info': calib_info,
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
        """HID线程"""
        logger.info("HID线程启动")

        while is_running() and self._running:
            try:
                dx, dy, should_fire = self.aim_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            # 合并同批所有后续瞄准指令
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
            self.state.update_mouse(total_dx, total_dy)

            # 自动开火
            if any(fire_flags):
                self.mouse.click("left")

    def _handle_key_input(self, key: int) -> bool:
        """
        处理键盘输入
        
        Returns:
            False: 退出程序
            True: 继续运行
        """
        # 准星校准模式
        if self.aiming.calibrator.is_active():
            if not self.aiming.calibrator.handle_key(key):
                # 校准结束
                pass
            return True

        # 常规按键
        if key == ord('q') or key == 27:  # Q 或 ESC
            self._running = False
            return False

        elif key == ord('f') or key == 0x700000:  # F1
            # 切换武器
            self.current_weapon_idx = (self.current_weapon_idx + 1) % len(self.weapon_list)
            weapon = self.weapon_list[self.current_weapon_idx]
            self.aiming.set_weapon(weapon)
            logger.info(f"切换武器: {weapon}")

        elif key == ord('g') or key == 0x710000:  # F2
            # 开关压枪
            RECOIL_CFG.enabled = not RECOIL_CFG.enabled
            logger.info(f"压枪补偿: {'启用' if RECOIL_CFG.enabled else '禁用'}")

        elif key == 0x720000:  # F3
            # 进入手动校准模式
            if CALIB_CFG.enabled:
                self.aiming.calibrator.start_manual_calibration()
                logger.info("进入手动准星校准模式")

        elif key == 0x730000:  # F4
            # 自动校准 (需要vis_data)
            logger.info("请确保准星可见，按Enter确认自动校准")

        elif key == 0x740000:  # F5
            # 切换目标选择策略
            strategies = ["composite", "nearest", "threat", "orientation"]
            current = strategies.index(TARGET_PRIO_CFG.strategy) if TARGET_PRIO_CFG.strategy in strategies else 0
            TARGET_PRIO_CFG.strategy = strategies[(current + 1) % len(strategies)]
            logger.info(f"目标策略: {TARGET_PRIO_CFG.strategy}")

        elif key == ord('=') or key == ord('+'):
            # 增加灵敏度
            MOUSE_CFG.sensitivity_x = min(3.0, MOUSE_CFG.sensitivity_x + 0.1)
            MOUSE_CFG.sensitivity_y = min(3.0, MOUSE_CFG.sensitivity_y + 0.1)
            logger.info(f"灵敏度: {MOUSE_CFG.sensitivity_x:.2f}")

        elif key == ord('-') or key == ord('_'):
            # 降低灵敏度
            MOUSE_CFG.sensitivity_x = max(0.1, MOUSE_CFG.sensitivity_x - 0.1)
            MOUSE_CFG.sensitivity_y = max(0.1, MOUSE_CFG.sensitivity_y - 0.1)
            logger.info(f"灵敏度: {MOUSE_CFG.sensitivity_x:.2f}")

        elif key == ord('1'):
            AIMING_CFG.priority_keypoints = [0, 1, 2]  # 头部
            logger.info("瞄准部位: 头部")
        elif key == ord('2'):
            AIMING_CFG.priority_keypoints = [5, 6, 11, 12]  # 躯干
            logger.info("瞄准部位: 躯干")

        return True

    def run(self):
        """主线程：调试显示 + 视频录制 + 性能监控 + 键盘交互"""
        logger.info("主循环启动（可视化线程）")

        wait_start = time.time()
        while self.vis_queue.empty() and (time.time() - wait_start) < 10.0:
            time.sleep(0.01)

        if self.vis_queue.empty():
            logger.warning("可视化队列无数据")

        frame_interval = 1.0 / SYS_CFG.target_fps
        last_frame_time = time.time()

        while is_running() and self._running:
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
            recoil_info = vis_data.get('recoil_info', {})
            calib_info = vis_data.get('calib_info', {})

            if SYS_CFG.show_debug_window:
                # 绘制增强版调试信息
                vis_frame = draw_debug_info(
                    frame, persons, target_person, aim_point, total_latency,
                    show_orientation=BODY_ORI_CFG.enabled,
                    calibrator_info=calib_info if self.aiming.calibrator.is_active() else None,
                    recoil_info=recoil_info
                )

                # 叠加校准UI (激活时)
                if self.aiming.calibrator.is_active():
                    vis_frame = self.aiming.calibrator.draw_calibration_ui(vis_frame)

                # 性能信息
                perf_text = self.monitor.get_summary()
                cv2.putText(vis_frame, perf_text, (10, vis_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 控制提示
                control_text = "F1:Weapon F2:Recoil F3:Calib F4:AutoCalib F5:Strategy +/-:Sens 1/2:Aim Q:Quit"
                cv2.putText(vis_frame, control_text, (10, vis_frame.shape[0] - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

                # 缩放显示
                if SYS_CFG.debug_scale != 1.0:
                    dw = int(vis_frame.shape[1] * SYS_CFG.debug_scale)
                    dh = int(vis_frame.shape[0] * SYS_CFG.debug_scale)
                    vis_frame = cv2.resize(vis_frame, (dw, dh))

                cv2.imshow("RTMO AimBot Enhanced", vis_frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    if not self._handle_key_input(key):
                        break

            # 录制调试视频
            if self.recorder is not None:
                vis_frame = draw_debug_info(
                    frame, persons, target_person, aim_point, total_latency
                )
                self.recorder.write(vis_frame)

            # 帧率控制
            elapsed = time.time() - last_frame_time
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)
            last_frame_time = time.time()

        self.shutdown()

    def shutdown(self):
        """关闭所有资源"""
        logger.info("正在关闭流水线...")
        self._running = False

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
    parser = argparse.ArgumentParser(
        description="RTMO AimBot Enhanced for Jetson AGX Xavier + ESP32-S3"
    )
    parser.add_argument("--engine", type=str, default=MODEL_CFG.engine_path,
                       help="TensorRT 引擎路径")
    parser.add_argument("--dummy-mouse", action="store_true",
                       help="使用虚拟鼠标模式")
    parser.add_argument("--dummy-esp32", action="store_true",
                       help="使用虚拟ESP32模式 (本地测试)")
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
    parser.add_argument("--weapon", type=str, default="rifle",
                       help="默认武器 (rifle/smg/lmg)")
    parser.add_argument("--mouse-mode", type=str, default=None,
                       choices=["esp32", "local", "dummy"],
                       help="鼠标控制模式")
    parser.add_argument("--serial-port", type=str, default=None,
                       help="ESP32串口路径 (如 /dev/ttyACM0)")

    args = parser.parse_args()

    # 应用命令行参数
    SYS_CFG.show_debug_window = args.debug or SYS_CFG.show_debug_window
    SYS_CFG.save_debug_video = args.record or SYS_CFG.save_debug_video
    CAPTURE_CFG.device = args.device
    MODEL_CFG.conf_thresh = args.conf
    MOUSE_CFG.sensitivity_x = args.sensitivity
    MOUSE_CFG.sensitivity_y = args.sensitivity

    if args.mouse_mode:
        MOUSE_CFG.mouse_mode = args.mouse_mode

    if args.serial_port:
        ESP32_CFG.serial_port = args.serial_port

    RECOIL_CFG.current_weapon = args.weapon

    setup_logging(SYS_CFG.log_level)

    # 检查权限
    if os.geteuid() != 0 and not args.dummy_mouse and args.mouse_mode == "local":
        logger.warning("未以root运行，HID设备可能需要额外权限")
        logger.warning("建议: sudo python3 main.py")

    # 启动
    pipeline = AimBotPipelineEnhanced(
        args.engine,
        dummy_mouse=args.dummy_mouse,
        dummy_esp32=args.dummy_esp32
    )
    pipeline.start()
    pipeline.run()


if __name__ == "__main__":
    main()
