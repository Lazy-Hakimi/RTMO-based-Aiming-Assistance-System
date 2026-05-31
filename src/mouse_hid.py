"""
Linux HID 鼠标控制器 (多线程高频率版)
通过 uinput 内核模块创建虚拟鼠标设备
支持相对移动和按键事件，事件驱动发送
"""
import os
import time
import logging
import struct
import fcntl
import threading
import queue
from typing import Tuple, Optional

from src.config import MOUSE_CFG

logger = logging.getLogger(__name__)

# Linux input event constants
EV_SYN = 0x00
EV_KEY = 0x01
EV_REL = 0x02

REL_X = 0x00
REL_Y = 0x01
REL_WHEEL = 0x08
REL_HWHEEL = 0x06

BTN_LEFT = 0x110
BTN_RIGHT = 0x111
BTN_MIDDLE = 0x112

# uinput constants
UI_DEV_CREATE = 0x5501
UI_DEV_DESTROY = 0x5502
UI_SET_RELBIT = 0x4551
UI_SET_KEYBIT = 0x4552
UI_SET_EVBIT = 0x4553

# input_event 结构体
INPUT_EVENT_FORMAT = "llHHi"
INPUT_EVENT_SIZE = struct.calcsize(INPUT_EVENT_FORMAT)

UINPUT_MAX_NAME_SIZE = 80


class HIDMouseController:
    """
    Linux uinput 虚拟鼠标控制器 (事件驱动，支持 1000Hz 等效频率)
    创建 /dev/uinput 设备，模拟鼠标相对移动
    """

    def __init__(self, device_path: str = "/dev/uinput"):
        self.device_path = device_path
        self.fd = None
        self._running = False
        self._move_queue = queue.Queue(maxsize=20)  # 事件队列，有界
        self._send_thread = None

        # 打开设备
        self._open_device()

    def _open_device(self):
        """打开并配置 uinput 设备"""
        try:
            self.fd = os.open(self.device_path, os.O_WRONLY | os.O_NONBLOCK)
        except PermissionError:
            logger.error(f"权限不足，无法打开 {self.device_path}")
            logger.error("请运行: sudo chmod 666 /dev/uinput")
            raise
        except FileNotFoundError:
            logger.error(f"设备不存在: {self.device_path}")
            logger.error("请加载 uinput 模块: sudo modprobe uinput")
            raise

        # 配置事件类型
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_SYN)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
        fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY)

        # 配置相对轴
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_X)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_Y)
        fcntl.ioctl(self.fd, UI_SET_RELBIT, REL_WHEEL)

        # 配置按键
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_LEFT)
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_RIGHT)
        fcntl.ioctl(self.fd, UI_SET_KEYBIT, BTN_MIDDLE)

        # 创建设备
        bus = 0x03  # BUS_USB
        vendor = 0x1234
        product = 0x5678
        version = 1

        name = b"RTMO-AimBot Virtual Mouse\x00"
        name = name.ljust(UINPUT_MAX_NAME_SIZE, b'\x00')

        setup = struct.pack("HHHH", bus, vendor, product, version) + name + struct.pack("I", 0)

        try:
            fcntl.ioctl(self.fd, UI_DEV_CREATE, setup)
        except OSError as e:
            # 回退到旧版 uinput 接口
            logger.warning(f"uinput_setup 失败，尝试旧接口: {e}")
            self._setup_legacy()

        time.sleep(0.1)  # 等待设备创建
        self._running = True

        # 启动事件驱动发送线程
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

        logger.info("HID 鼠标设备创建成功 (事件驱动模式)")

    def _setup_legacy(self):
        """旧版 uinput 接口 (兼容老内核)"""
        name = b"RTMO-AimBot Virtual Mouse\x00"
        name = name.ljust(UINPUT_MAX_NAME_SIZE, b'\x00')

        bus = 0x03
        vendor = 0x1234
        product = 0x5678
        version = 1

        device_data = name
        device_data += struct.pack("HHHH", bus, vendor, product, version)
        device_data += struct.pack("i", 0)  # ff_effects_max
        device_data += b'\x00' * (64 * 4)  # absmax, absmin, absfuzz, absflat

        os.write(self.fd, device_data)
        fcntl.ioctl(self.fd, UI_DEV_CREATE)

    def _send_loop(self):
        """事件驱动发送线程：阻塞等待，有数据立即发送并合并"""
        while self._running:
            try:
                dx, dy = self._move_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            # 合并同批所有后续移动事件（减少系统调用）
            total_dx, total_dy = dx, dy
            merge_count = 1
            while not self._move_queue.empty():
                try:
                    dx2, dy2 = self._move_queue.get_nowait()
                    total_dx += dx2
                    total_dy += dy2
                    merge_count += 1
                except queue.Empty:
                    break

            self._send_move_raw(total_dx, total_dy)

    def _write_event(self, type_: int, code: int, value: int):
        """写入单个 input_event"""
        event = struct.pack(INPUT_EVENT_FORMAT, 0, 0, type_, code, value)
        try:
            os.write(self.fd, event)
        except BlockingIOError:
            # 非阻塞模式下缓冲区满，静默丢弃
            pass

    def _send_move_raw(self, dx: int, dy: int):
        """直接发送原始鼠标移动事件 (内部使用)"""
        if dx != 0:
            self._write_event(EV_REL, REL_X, int(dx))
        if dy != 0:
            self._write_event(EV_REL, REL_Y, int(dy))
        self._write_event(EV_SYN, 0, 0)

    def move(self, dx: float, dy: float):
        """
        发送鼠标相对移动 (非阻塞，队列满则丢弃最旧)
        Args:
            dx, dy: 像素偏移量 (会被转换为鼠标单位)
        """
        if not MOUSE_CFG.enabled or self.fd is None:
            return

        dx_int = int(round(dx))
        dy_int = int(round(dy))

        if dx_int == 0 and dy_int == 0:
            return

        try:
            self._move_queue.put_nowait((dx_int, dy_int))
        except queue.Full:
            # 丢弃最旧的事件，放入最新
            try:
                _ = self._move_queue.get_nowait()
                self._move_queue.put_nowait((dx_int, dy_int))
            except queue.Empty:
                pass

    def click(self, button: str = "left"):
        """模拟鼠标点击"""
        if not MOUSE_CFG.enabled or self.fd is None:
            return

        btn_map = {
            "left": BTN_LEFT,
            "right": BTN_RIGHT,
            "middle": BTN_MIDDLE
        }
        btn = btn_map.get(button, BTN_LEFT)

        self._write_event(EV_KEY, btn, 1)  # press
        self._write_event(EV_SYN, 0, 0)
        time.sleep(0.05)
        self._write_event(EV_KEY, btn, 0)  # release
        self._write_event(EV_SYN, 0, 0)

    def press(self, button: str = "left"):
        """按下鼠标按键 (不释放)"""
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        btn_map = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}
        self._write_event(EV_KEY, btn_map.get(button, BTN_LEFT), 1)
        self._write_event(EV_SYN, 0, 0)

    def release(self, button: str = "left"):
        """释放鼠标按键"""
        if not MOUSE_CFG.enabled or self.fd is None:
            return
        btn_map = {"left": BTN_LEFT, "right": BTN_RIGHT, "middle": BTN_MIDDLE}
        self._write_event(EV_KEY, btn_map.get(button, BTN_LEFT), 0)
        self._write_event(EV_SYN, 0, 0)

    def close(self):
        """关闭设备"""
        self._running = False
        if self._send_thread:
            self._send_thread.join(timeout=1.0)

        if self.fd is not None:
            try:
                fcntl.ioctl(self.fd, UI_DEV_DESTROY)
                os.close(self.fd)
            except Exception as e:
                logger.warning(f"关闭设备时出错: {e}")
            self.fd = None
            logger.info("HID 鼠标设备已关闭")


class MouseControllerDummy:
    """调试用的虚拟鼠标控制器 (不实际操作鼠标)"""

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
                    f"Total offset: ({self.total_dx:.1f}, {self.total_dy:.1f})")


def create_mouse_controller(dummy: bool = False):
    """工厂函数：创建鼠标控制器"""
    if dummy or not MOUSE_CFG.enabled:
        return MouseControllerDummy()
    try:
        return HIDMouseController(MOUSE_CFG.uinput_device)
    except Exception as e:
        logger.error(f"创建 HID 控制器失败: {e}")
        logger.warning("回退到虚拟模式")
        return MouseControllerDummy()
