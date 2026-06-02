"""
视频采集模块
支持 V4L2 HDMI采集卡、MJPEG/YUYV 解码
Jetson 硬件加速解码 (NVDEC/JPEG)
"""
import os
import logging
import time
import threading
from typing import Tuple, Optional
from collections import deque

import numpy as np
import cv2

from src.config import CAPTURE_CFG, SYS_CFG

logger = logging.getLogger(__name__)


class VideoCapture:
    """
    视频采集器 (线程安全)
    使用 OpenCV VideoCapture 后端，支持 V4L2
    """

    def __init__(self, device: str = "/dev/video0"):
        self.device = device
        self.cap = None
        self.frame_width = CAPTURE_CFG.capture_width
        self.frame_height = CAPTURE_CFG.capture_height
        self.fps = CAPTURE_CFG.capture_fps

        # 性能统计
        self.frame_times = deque(maxlen=30)
        self.last_frame_time = 0
        self.actual_fps = 0.0

        # 最新帧缓存 (用于异步读取)
        self._latest_frame = None
        self._frame_timestamp = 0

        # 线程锁
        self._lock = threading.Lock()

        self._open()

    def _open(self):
        """打开视频设备"""
        logger.info(f"正在打开视频设备: {self.device}")

        # 尝试 V4L2 后端
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            # 回退到默认后端
            self.cap = cv2.VideoCapture(self.device)

        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频设备: {self.device}")

        # 设置采集参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # 设置像素格式
        if CAPTURE_CFG.pixel_format.upper() == "MJPEG":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        elif CAPTURE_CFG.pixel_format.upper() == "YUYV":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

        # 设置缓冲区大小 (减少延迟)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAPTURE_CFG.buffer_count)

        # 读取实际参数
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(f"采集参数: {actual_w}x{actual_h} @ {actual_fps:.1f}fps")
        logger.info(f"像素格式: {CAPTURE_CFG.pixel_format}")

        # 预热 (丢弃前几帧，让采集卡稳定)
        for _ in range(5):
            self.cap.read()

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        读取一帧 (线程安全)
        Returns:
            (success, frame)  frame 为 BGR 格式
        """
        if self.cap is None or not self.cap.isOpened():
            return False, None

        t0 = time.time()
        ret, frame = self.cap.read()
        t1 = time.time()

        if ret:
            with self._lock:
                self.frame_times.append(t1 - t0)
                if len(self.frame_times) > 10:
                    self.actual_fps = len(self.frame_times) / sum(self.frame_times)
                self._latest_frame = frame
                self._frame_timestamp = t1
        else:
            logger.warning("帧读取失败")

        return ret, frame

    def get_latest(self) -> Optional[np.ndarray]:
        """获取最新缓存帧 (线程安全)"""
        with self._lock:
            return self._latest_frame

    def get_fps(self) -> float:
        """获取实际采集帧率 (线程安全)"""
        with self._lock:
            return self.actual_fps

    def release(self):
        """释放资源"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info("视频采集已释放")


class VideoCaptureGStreamer(VideoCapture):
    """
    使用 GStreamer Pipeline 的采集器 (Jetson 硬件加速)
    支持 NVDEC 硬件解码 MJPEG/H.264
    """

    def __init__(self, device: str = "/dev/video0"):
        # 显式初始化锁 (本类不调用父类__init__)
        self._lock = threading.Lock()
        self.device = device
        self.cap = None
        self.frame_width = CAPTURE_CFG.capture_width
        self.frame_height = CAPTURE_CFG.capture_height
        self.fps = CAPTURE_CFG.capture_fps
        self.frame_times = deque(maxlen=30)
        self.actual_fps = 0.0
        self._latest_frame = None
        self._frame_timestamp = 0
        self._open_gstreamer()

    def _open_gstreamer(self):
        """构建 GStreamer Pipeline"""
        w, h = self.frame_width, self.frame_height
        fps = self.fps

        # 针对 Jetson 优化的 GStreamer Pipeline
        # 使用 nvjpegdec 或 nvv4l2decoder 进行硬件解码
        if CAPTURE_CFG.pixel_format.upper() == "MJPEG":
            # MJPEG 硬件解码 Pipeline
            pipeline = (
                f"v4l2src device={self.device} io-mode=2 ! "
                f"image/jpeg, width={w}, height={h}, framerate={fps}/1 ! "
                f"nvjpegdec ! "
                f"video/x-raw, format=NV12 ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )
        elif CAPTURE_CFG.pixel_format.upper() == "YUYV":
            # YUYV 直接采集
            pipeline = (
                f"v4l2src device={self.device} io-mode=2 ! "
                f"video/x-raw, format=YUY2, width={w}, height={h}, framerate={fps}/1 ! "
                f"nvvidconv ! "
                f"video/x-raw, format=BGRx ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )
        else:
            # 默认
            pipeline = (
                f"v4l2src device={self.device} ! "
                f"video/x-raw, width={w}, height={h} ! "
                f"videoconvert ! "
                f"video/x-raw, format=BGR ! "
                f"appsink drop=true max-buffers=1"
            )

        logger.info(f"GStreamer Pipeline:\n{pipeline}")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            logger.warning("GStreamer 打开失败，回退到标准 V4L2")
            super()._open()
            return

        # 预热
        for _ in range(5):
            self.cap.read()

        logger.info("GStreamer 硬件加速采集已启动")


def create_capture(use_gstreamer: bool = True) -> VideoCapture:
    """工厂函数：创建采集器"""
    if use_gstreamer and CAPTURE_CFG.use_hw_decode:
        try:
            return VideoCaptureGStreamer(CAPTURE_CFG.device)
        except Exception as e:
            logger.warning(f"GStreamer 采集失败: {e}")
    return VideoCapture(CAPTURE_CFG.device)
