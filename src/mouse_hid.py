"""
鼠标HID控制模块 (增强版)
支持三种模式:
1. esp32 - 通过ESP32-S3桥接发送鼠标指令 (推荐用于Jetson Xavier)
2. local  - 本地uinput直接控制 (仅在Xavier直接连游戏主机时使用)
3. dummy  - 虚拟模式 (调试)
"""
import os
import time
import logging
import struct
import fcntl
import threading
import queue
from typing import Tuple, Optional

from src.config import MOUSE_CFG, ESP32_CFG

logger = logging.getLogger(__name__)

# Linux input event constants (local模式用)
EV_SYN = 0x00
EV_KEY = 0x01
EV_REL = 0x02
REL_X = 0x00
REL_Y = 0x01
REL_WHEEL = 0x08
BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BTN_MIDDLE = 0x112
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_RELBIT = 0x4551
UI_SET_KEYBIT = 0x4552
UI_SET_EVBIT = 0x4553
INPUT_EVENT_FORMAT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FORMAT)
UINPUT_MAX_NAME_SIZE = 80


class MouseControllerBase:
    """鼠标控制器基类"""

    def move(self, dx: float, dy: float):
        raise NotImplementedError

    def click(self, button: str = "left"):
        raise NotImplementedError

    def press(self, button: str = "left"):
        raise NotImplementedError

    def release(self, button: str = "left"):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def is_ready(self) -> bool:
        return True


class ESP32MouseController(MouseControllerBase):
    """
    ESP32-S3 桥接鼠标控制器
    通过串口/USB向ESP32-S3发送鼠标指令，由ESP32模拟HID鼠标
    """

    def __init__(self):
        # 延迟导入以避免循环依赖
        from src.esp32_bridge import ESP32Bridge
        self.bridge = ESP32Bridge(ESP32_CFG)
        self.bridge.start()

        # 按键映射
        self._btn_map = {
            "left": 0,
            "right": 1,
            "middle": 2,
        }

        logger.info("ESP32-S3鼠标控制器已创建")

    def move(self, dx: float, dy: float):
        """发送鼠标移动指令到ESP32-S3"""
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return
        self.bridge.send_mouse_move(dx, dy)

    def click(self, button: str = "left"):
        """发送鼠标点击指令"""
        btn_code = self._btn_map.get(button, 0)
        self.bridge.send_mouse_click(btn_code)

    def press(self, button: str = "left"):
        """按下 (暂不支持长按，需要协议扩展)"""
        logger.debug(f"ESP32模式暂不支持长按: {button}")

    def release(self, button: str = "left"):
        """释放"""
        pass

    def close(self):
        """关闭桥接"""
        self.bridge.stop()

    def is_ready(self) -> bool:
        return self.bridge.is_connected()

    def get_status(self) -> str:
        return self.bridge.get_stats()


class HIDMouseController(MouseControllerBase):
    """本地uinput HID鼠标控制器 (原功能保留)"""

    def __init__(self, device_path: str = "/dev/uinput"):
        self.device_path = device_path
        self.fd = None
        self._running = False
        self._move_queue = queue.Queue(maxsize=20)
        self._send_thread = None
        self._open_device()

    def _open_device(self):
        try:
            self.fd = os.open(self.device_path, os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError:
            logger.error(f"权限不足: {self.device_path}")
            raise
        except FileNotFoundError:
            logger.error(f"设备不存在: {self.device_path}")
            raise

        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_SYN)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_X)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_Y)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_WHEEL)
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_LEFT)
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_RIGHT)
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_MIDDLE)

        bus = 0x03
        vendor = 0x1234
        product = 0x5678
        version = 1
        name = b"RTMO-AimBot Virtual Mouse\x00"
        name = name.ljust(UINPUT_MAX_NAME_SIZE, b'\x00')
        setup = struct.pack("HHHH", bus, vendor, product, version) + name + struct.pack("I", 0)

        try:
            fcntl.ioctl(self.fd, UI_DEV_CREATE, setup)
        except OSError:
            self._setup_legacy()

        time.sleep(0.1)
        self._running = True
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()
        logger.info("本地HID鼠标设备创建成功")

    def _setup_legacy(self):
        name = b"RTMO-AimBot Virtual Mouse\x00"
        name = name.ljust(UINPUT_MAX_NAME_SIZE, b'\x00')
        bus = 0x03
        vendor = 0x1234
        product = 0x5678
        version = 1
        device_data = name + struct.pack("HHHH", bus, vendor, product, version)
        device_data += struct.pack("i", 0)
        device_data += b'\x00' * (64 * 4)
        os.write(self.fd, device_data)
        fcntl.ioctl(self.fd, UI_DEV_CREATE)

    def _send_loop(self):
        while self._running:
            try:
                dx, dy = self._move_queue.get(timeout=0.001)
            except queue.Empty:
                continue
            total_dx, total_dy = dx, dy
            while not self._move_queue.empty():
                try:
                    dx2, dy2 = self._move_queue.get_nowait()
                    total_dx += dx2
                    total_dy += dy2
                except queue.Empty:
                    break
            self._send_move_raw(total_dx, total_dy)

    def _write_event(self, type_: int, code: int, value: int):
        event = struct.pack(INPUT_EVENT_FORMAT, 0, 0, type_, code, value)
        try:
            os.write(self.fd, event)
        except BlockingIOError:
            pass

    def _send_move_raw(self, dx: int, dy: int):
        if dx != 0:
            self._write_event(EV_REL, REL_X, int(dx))
        if dy != 0:
            self._write_event(EV_REL, REL_Y, int(dy))
        self._write_event(EV_SYN, 0, 0)

    def move(self, dx: float, dy: float):
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        dx_int = int(round(dx))
        dy_int = int(round(dy))
        if dx_int == 0 and dy_int == 0:
            return
        try:
            self._move_queue.put_nowait((dx_int, dy_int))
        except queue.Full:
            try:
                _ = self._move_queue.get_nowait()
                self._move_queue.put_nowait((dx_int, dy_int))
            except queue.Empty:
                pass

    def click(self, button: str = "left"):
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        btn_map = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}
        btn = btn_map.get(button, BTN_LEFT)
        self._write_event(EV_KEY, btn, 1)
        self._write_event(EV_SYN, 0, 0)
        time.sleep(0.05)
        self._write_event(EV_KEY, btn, 0)
        self._write_event(EV_SYN, 0, 0)

    def press(self, button: str = "left"):
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        btn_map = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}
        self._write_event(EV_KEY, btn_map.get(button, BTN_LEFT), 1)
        self._write_event(EV_SYN, 0, 0)

    def release(self, button: str = "left"):
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        btn_map = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}
        self._write_event(EV_KEY, btn_map.get(button, BTN_LEFT), 0)
        self._write_event(EV_SYN, 0, 0)

    def close(self):
        self._running = False
        if self._send_thread:
            self._send_thread.join(timeout=1.0)
        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, UI_DEV_DESTROY)
                os.close(self.fd)
            except Exception as e:
                logger.warning(f"关闭设备出错: {e}")
            self.fd = None
            logger.info("本地HID鼠标设备已关闭")


class MouseControllerDummy(MouseControllerBase):
    """虚拟鼠标控制器 (调试)"""

    def __init__(self):
        self.total_dx = 0
        self.total_dy = 0
        self.move_count = 0

    def move(self, dx: float, dy: float):
        self.total_dx += dx
        self.total_dy += dy
        self.move_count += 1
        logger.debug(f"[DUMMY] Move: dx={dx:.1f}, dy={dy:.1f}")

    def click(self, button: str = "left"):
        logger.debug(f"[DUMMY] Click: {button}")

    def press(self, button: str = "left"):
        logger.debug(f"[DUMMY] Press: {button}")

    def release(self, button: str = "left"):
        logger.debug(f"[DUMMY] Release: {button}")

    def close(self):
        logger.info(f"[DUMMY] Total moves: {self.move_count}, "
                    f"offset: ({self.total_dx:.1f}, {self.total_dy:.1f})")


def create_mouse_controller(mode: str = None, dummy: bool = False) -> MouseControllerBase:
    """
    工厂函数：创建鼠标控制器
    
    Args:
        mode: 控制模式 "esp32"|"local"|"dummy"|None(从配置读取)
        dummy: 是否强制使用虚拟模式
    """
    if dummy:
        return MouseControllerDummy()

    if mode is None:
        mode = MOUSE_CFG.mouse_mode

    if mode == "esp32":
        try:
            return ESP32MouseController()
        except Exception as e:
            logger.error(f"创建ESP32控制器失败: {e}")
            logger.warning("回退到虚拟模式")
            return MouseControllerDummy()

    elif mode == "local":
        try:
            return HIDMouseController(MOUSE_CFG.uinput_device)
        except Exception as e:
            logger.error(f"创建本地HID控制器失败: {e}")
            return MouseControllerDummy()

    else:  # dummy
        return MouseControllerDummy()
