"""
工具函数：性能监控、日志、可视化
"""
import os
import time
import logging
import signal
import threading
from typing import Optional
from collections import deque
from dataclasses import dataclass

import numpy as np
import cv2

from src.config import SYS_CFG


# 全局运行标志 (用于信号处理)
_running = True

def signal_handler(signum, frame):
    global _running
    _running = False
    logging.info(f"收到信号 {signum}，正在退出...")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def is_running() -> bool:
    return _running


@dataclass
class PerformanceMetrics:
    """性能指标"""
    capture_fps: float = 0.0
    inference_ms: float = 0.0
    preprocess_ms: float = 0.0
    postprocess_ms: float = 0.0
    total_latency_ms: float = 0.0
    mouse_move_ms: float = 0.0
    persons_detected: int = 0
    target_locked: bool = False


class PerformanceMonitor:
    """线程安全性能监控器"""

    def __init__(self, window_size: int = 30):
        self._lock = threading.Lock()
        self.window = window_size
        self.capture_times = deque(maxlen=window_size)
        self.infer_times = deque(maxlen=window_size)
        self.preprocess_times = deque(maxlen=window_size)
        self.postprocess_times = deque(maxlen=window_size)
        self.total_times = deque(maxlen=window_size)
        self.frame_count = 0
        self.start_time = time.time()

    def record(self, metrics: PerformanceMetrics):
        with self._lock:
            self.capture_times.append(metrics.capture_fps)
            self.infer_times.append(metrics.inference_ms)
            self.preprocess_times.append(metrics.preprocess_ms)
            self.postprocess_times.append(metrics.postprocess_ms)
            self.total_times.append(metrics.total_latency_ms)
            self.frame_count += 1

    def get_summary(self) -> str:
        with self._lock:
            if not self.infer_times:
                return "暂无数据"

            elapsed = time.time() - self.start_time
            avg_fps = self.frame_count / elapsed if elapsed > 0 else 0

            return (
                f"FPS: {avg_fps:.1f} | "
                f"Capture: {np.mean(self.capture_times):.1f}fps | "
                f"Infer: {np.mean(self.infer_times):.1f}ms | "
                f"Pre: {np.mean(self.preprocess_times):.1f}ms | "
                f"Post: {np.mean(self.postprocess_times):.1f}ms | "
                f"Total: {np.mean(self.total_times):.1f}ms"
            )

    def print_stats(self):
        with self._lock:
            logging.info(self.get_summary())


class VideoRecorder:
    """调试视频录制器"""

    def __init__(self, output_path: str, fps: int = 30, resolution: tuple = (1920, 1080)):
        self.output_path = output_path
        self.fps = fps
        self.resolution = resolution
        self.writer = None
        self._ensure_dir()

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.output_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

    def start(self):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频文件: {self.output_path}")
        logging.info(f"开始录制调试视频: {self.output_path}")

    def write(self, frame: np.ndarray):
        if self.writer is not None:
            # 确保尺寸匹配
            if frame.shape[:2] != (self.resolution[1], self.resolution[0]):
                frame = cv2.resize(frame, self.resolution)
            self.writer.write(frame)

    def stop(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            logging.info("调试视频录制已停止")


def setup_logging(level: str = "INFO"):
    """配置日志"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S"
    )
