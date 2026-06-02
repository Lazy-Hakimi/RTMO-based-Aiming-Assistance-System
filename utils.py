"""
ESP32-S3 通信桥接模块 (ESP32 Bridge)
Jetson AGX Xavier (推理端) 通过此模块向 ESP32-S3 (HID设备端) 发送鼠标控制指令

通信协议:
- 物理层: UART (Serial) / USB-CDC / UDP (WiFi)
- 数据层: 二进制协议 (高效) 或 JSON文本协议 (易调试)
- 应用层: 鼠标移动、按键、状态查询

二进制协议格式 (v1):
    包头:     0xAA 0x55 (2 bytes, 固定)
    版本:     0x01 (1 byte)
    指令类型: 1 byte (见下)
    数据长度: 2 bytes (小端序)
    数据:     N bytes
    校验和:   1 byte (累加和)

指令类型:
    0x01 - 鼠标相对移动 (数据: dx(int16), dy(int16), buttons(uint8))
    0x02 - 鼠标按键 (数据: button(uint8), state(uint8: 0=release, 1=press))
    0x03 - 滚轮 (数据: vertical(int8), horizontal(int8))
    0x04 - 综合指令 (数据: dx(int16), dy(int16), buttons(uint8), wheel_v(int8))
    0x05 - 心跳 (数据: 无)
    0x06 - 状态查询 (数据: 无)
    0x07 - 配置设置 (数据: key(uint8), value(int16))
    0x10 - 状态响应 (ESP32 -> Xavier)
    0x11 - 错误报告 (ESP32 -> Xavier)
    0xFF - 批量指令头 (后接多条指令)

ESP32-S3 响应格式:
    包头:     0xBB 0x66
    版本:     0x01
    响应类型: 1 byte
    状态:     1 byte (0=OK, 1=Error, 2=Busy)
    数据长度: 2 bytes
    数据:     N bytes
    校验和:   1 byte
"""
import os
import time
import struct
import logging
import threading
import queue
from typing import Tuple, Optional, Dict, List
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入串口库
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    logger.warning("pyserial 未安装，串口功能不可用. 安装: pip install pyserial")

# 协议常量
HEADER_TX = b'\xAA\x55'      # 发送包头
HEADER_RX = b'\xBB\x66'      # 接收包头
PROTO_VERSION = 0x01

# 指令类型
CMD_MOUSE_MOVE = 0x01
CMD_MOUSE_BUTTON = 0x02
CMD_MOUSE_WHEEL = 0x03
CMD_MOUSE_COMBINED = 0x04
CMD_HEARTBEAT = 0x05
CMD_STATUS_QUERY = 0x06
CMD_CONFIG_SET = 0x07

# 响应类型
RESP_STATUS = 0x10
RESP_ERROR = 0x11
RESP_HEARTBEAT_ACK = 0x12

# 状态码
STATUS_OK = 0x00
STATUS_ERROR = 0x01
STATUS_BUSY = 0x02


class ESP32ProtocolEncoder:
    """ESP32通信协议编码器"""

    @staticmethod
    def encode_mouse_move(dx: int, dy: int, buttons: int = 0) -> bytes:
        """编码鼠标移动指令"""
        data = struct.pack('<hhb', int(dx), int(dy), buttons)
        return ESP32ProtocolEncoder._encode_frame(CMD_MOUSE_MOVE, data)

    @staticmethod
    def encode_mouse_button(button: int, pressed: bool) -> bytes:
        """编码鼠标按键指令"""
        data = struct.pack('<BB', button, 1 if pressed else 0)
        return ESP32ProtocolEncoder._encode_frame(CMD_MOUSE_BUTTON, data)

    @staticmethod
    def encode_mouse_wheel(vertical: int, horizontal: int = 0) -> bytes:
        """编码滚轮指令"""
        data = struct.pack('<bb', int(vertical), int(horizontal))
        return ESP32ProtocolEncoder._encode_frame(CMD_MOUSE_WHEEL, data)

    @staticmethod
    def encode_combined(dx: int, dy: int, buttons: int = 0,
                        wheel_v: int = 0) -> bytes:
        """编码综合鼠标指令 (移动+按键+滚轮)"""
        data = struct.pack('<hhb b', int(dx), int(dy), buttons, wheel_v)
        return ESP32ProtocolEncoder._encode_frame(CMD_MOUSE_COMBINED, data)

    @staticmethod
    def encode_heartbeat() -> bytes:
        """编码心跳指令"""
        return ESP32ProtocolEncoder._encode_frame(CMD_HEARTBEAT, b'')

    @staticmethod
    def encode_status_query() -> bytes:
        """编码状态查询指令"""
        return ESP32ProtocolEncoder._encode_frame(CMD_STATUS_QUERY, b'')

    @staticmethod
    def encode_config(key: int, value: int) -> bytes:
        """编码配置设置指令"""
        data = struct.pack('<Bh', key, value)
        return ESP32ProtocolEncoder._encode_frame(CMD_CONFIG_SET, data)

    @staticmethod
    def _encode_frame(cmd_type: int, data: bytes) -> bytes:
        """
        编码完整帧
        
        Format: [Header(2)] [Version(1)] [Cmd(1)] [Len(2)] [Data(N)] [Checksum(1)]
        """
        frame = bytearray()
        frame.extend(HEADER_TX)
        frame.append(PROTO_VERSION)
        frame.append(cmd_type)
        frame.extend(struct.pack('<H', len(data)))
        frame.extend(data)

        # 校验和 (累加和)
        checksum = sum(frame) & 0xFF
        frame.append(checksum)

        return bytes(frame)

    @staticmethod
    def calc_packet_size() -> int:
        """计算固定包头大小"""
        return 2 + 1 + 1 + 2  # Header + Version + Cmd + Len


class ESP32ProtocolDecoder:
    """ESP32通信协议解码器"""

    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data: bytes) -> List[Dict]:
        """
        接收数据并尝试解析完整帧
        
        Returns:
            解析出的消息列表
        """
        self._buffer.extend(data)
        messages = []

        while True:
            # 查找包头
            header_idx = self._buffer.find(HEADER_RX)
            if header_idx < 0:
                break

            # 检查是否有足够的数据
            # Header(2) + Version(1) + RespType(1) + Status(1) + Len(2) + Checksum(1) = 8
            min_size = 7
            if len(self._buffer) < header_idx + min_size:
                break

            # 解析长度 (在Status之后)
            msg_len = struct.unpack('<H', self._buffer[header_idx+5:header_idx+7])[0]
            total_len = min_size + msg_len + 1  # +1 for checksum

            if len(self._buffer) < header_idx + total_len:
                break

            # 提取完整帧
            frame = self._buffer[header_idx:header_idx + total_len]

            # 校验和验证
            calc_sum = sum(frame[:-1]) & 0xFF
            recv_sum = frame[-1]

            if calc_sum == recv_sum:
                msg = self._parse_frame(frame)
                if msg:
                    messages.append(msg)
            else:
                logger.warning(f"校验和错误: calc={calc_sum}, recv={recv_sum}")

            # 移除已处理的数据
            self._buffer = self._buffer[header_idx + total_len:]

        return messages

    def _parse_frame(self, frame: bytes) -> Optional[Dict]:
        """
        解析单帧数据 (响应帧格式)
        Format: [Header(2)] [Version(1)] [RespType(1)] [Status(1)] [Len(2)] [Data(N)] [Checksum(1)]
        """
        if len(frame) < 8:
            return None

        resp_type = frame[3]
        status = frame[4]
        data_len = struct.unpack('<H', frame[5:7])[0]
        payload = frame[7:7+data_len]

        if resp_type == RESP_STATUS:
            # 状态响应
            if len(payload) >= 4:
                return {
                    "type": "status",
                    "firmware_version": payload[0],
                    "mouse_hz": struct.unpack('<H', payload[1:3])[0],
                    "queue_depth": payload[3] if len(payload) > 3 else 0,
                    "status_code": status,
                }
        elif resp_type == RESP_ERROR:
            # 错误报告
            if len(payload) >= 1:
                return {
                    "type": "error",
                    "error_code": payload[0],
                    "status_code": status,
                    "error_msg": f"Error code {payload[0]}",
                }
        elif resp_type == RESP_HEARTBEAT_ACK:
            # 心跳响应
            return {"type": "heartbeat_ack", "status_code": status}

        return {"type": "unknown", "raw_type": resp_type, "status_code": status}


class ESP32Bridge:
    """
    ESP32-S3 通信桥接器
    管理推理端与ESP32-S3之间的通信
    """

    def __init__(self, cfg=None):
        from src.config import ESP32_CFG
        self.cfg = cfg or ESP32_CFG

        self.encoder = ESP32ProtocolEncoder()
        self.decoder = ESP32ProtocolDecoder()

        # 串口对象
        self._serial = None
        self._serial_lock = threading.Lock()

        # 运行状态
        self._running = False
        self._connected = False
        self._connect_thread = None
        self._send_thread = None
        self._recv_thread = None

        # 发送队列 (线程安全)
        self._send_queue = queue.Queue(maxsize=100)

        # 统计数据
        self._tx_count = 0
        self._rx_count = 0
        self._error_count = 0
        self._last_heartbeat = 0.0
        self._esp32_status = {}

        # 虚拟模式 (调试)
        self._dummy_accumulated_dx = 0.0
        self._dummy_accumulated_dy = 0.0

    def start(self) -> bool:
        """启动桥接器"""
        if self._running:
            return True

        self._running = True

        if self.cfg.dummy_esp32:
            logger.info("ESP32桥接器以虚拟模式启动")
            self._connected = True
            return True

        # 启动连接线程
        self._connect_thread = threading.Thread(target=self._connect_loop, daemon=True)
        self._connect_thread.start()

        # 启动发送线程
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._send_thread.start()

        # 启动接收线程
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()

        logger.info("ESP32桥接器已启动")
        return True

    def stop(self):
        """停止桥接器"""
        self._running = False
        self._connected = False

        for t in [self._connect_thread, self._send_thread, self._recv_thread]:
            if t and t.is_alive():
                t.join(timeout=1.0)

        if self._serial:
            try:
                self._serial.close()
            except Exception:
                pass
            self._serial = None

        logger.info("ESP32桥接器已停止")

    def _connect_loop(self):
        """连接保持循环"""
        reconnect_delay = self.cfg.reconnect_interval
        reconnect_count = 0

        while self._running:
            if not self._connected:
                if self._try_connect():
                    reconnect_count = 0
                    reconnect_delay = self.cfg.reconnect_interval
                else:
                    reconnect_count += 1
                    if reconnect_count > self.cfg.max_reconnect_attempts:
                        logger.error(f"达到最大重连次数 ({self.cfg.max_reconnect_attempts})，停止重连")
                        self._running = False
                        break

                    logger.warning(f"{reconnect_delay}秒后重连... (尝试 {reconnect_count}/{self.cfg.max_reconnect_attempts})")
                    time.sleep(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2, 30)  # 指数退避
            else:
                # 检查心跳超时
                if time.time() - self._last_heartbeat > self.cfg.heartbeat_timeout:
                    logger.warning("心跳超时，断开连接")
                    self._connected = False
                    if self._serial:
                        try:
                            self._serial.close()
                        except Exception:
                            pass
                        self._serial = None
                else:
                    # 定期发送心跳
                    if time.time() - self._last_heartbeat > self.cfg.heartbeat_interval:
                        self._queue_command(self.encoder.encode_heartbeat())

                time.sleep(0.1)

    def _try_connect(self) -> bool:
        """尝试连接ESP32-S3"""
        if not HAS_SERIAL:
            logger.error("pyserial未安装，无法连接串口")
            return False

        try:
            port = self.cfg.serial_port

            # 自动查找ESP32-S3端口
            if port == "auto":
                port = self._find_esp32_port()
                if port is None:
                    logger.warning("未找到ESP32-S3串口")
                    return False

            logger.info(f"正在连接ESP32-S3: {port} @ {self.cfg.serial_baudrate}bps")

            self._serial = serial.Serial(
                port=port,
                baudrate=self.cfg.serial_baudrate,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=self.cfg.serial_timeout,
                write_timeout=0.001,
            )

            # 等待设备就绪
            time.sleep(0.5)

            # 发送状态查询验证连接
            self._serial.write(self.encoder.encode_status_query())
            self._serial.flush()

            # 等待响应
            start = time.time()
            while time.time() - start < 2.0:
                if self._serial.in_waiting > 0:
                    data = self._serial.read(self._serial.in_waiting)
                    msgs = self.decoder.feed(data)
                    if msgs:
                        self._connected = True
                        self._last_heartbeat = time.time()
                        logger.info(f"ESP32-S3连接成功: {msgs[0]}")
                        return True
                time.sleep(0.1)

            # 未收到响应但串口已打开
            self._connected = True
            self._last_heartbeat = time.time()
            logger.info("ESP32-S3串口已打开 (未收到状态响应)")
            return True

        except Exception as e:
            logger.warning(f"连接失败: {e}")
            return False

    def _find_esp32_port(self) -> Optional[str]:
        """自动查找ESP32-S3串口"""
        if not HAS_SERIAL:
            return None

        ports = serial.tools.list_ports.comports()

        for p in ports:
            # ESP32-S3的典型描述
            desc_lower = (p.description or "").lower()
            hwid_lower = (p.hwid or "").lower()

            if any(k in desc_lower for k in ['esp32', 'usb jtag', 'cp210', 'ch340']):
                logger.info(f"找到可能的ESP32端口: {p.device} ({p.description})")
                return p.device

        # 返回第一个可用的USB串口
        for p in ports:
            if p.device.startswith('/dev/ttyACM') or p.device.startswith('/dev/ttyUSB'):
                return p.device

        return None

    def _send_loop(self):
        """发送循环"""
        while self._running:
            if not self._connected or self._serial is None:
                time.sleep(0.01)
                continue

            try:
                cmd = self._send_queue.get(timeout=0.001)
            except queue.Empty:
                continue

            try:
                with self._serial_lock:
                    self._serial.write(cmd)
                    self._serial.flush()
                    self._tx_count += 1
            except Exception as e:
                self._error_count += 1
                logger.warning(f"发送失败: {e}")
                self._connected = False

    def _recv_loop(self):
        """接收循环"""
        while self._running:
            if not self._connected or self._serial is None:
                time.sleep(0.01)
                continue

            try:
                with self._serial_lock:
                    available = self._serial.in_waiting
                    if available > 0:
                        data = self._serial.read(min(available, 256))
                        self._rx_count += len(data)

                        msgs = self.decoder.feed(data)
                        for msg in msgs:
                            self._handle_message(msg)
            except Exception as e:
                logger.warning(f"接收错误: {e}")

            time.sleep(0.001)

    def _handle_message(self, msg: Dict):
        """处理来自ESP32的消息"""
        msg_type = msg.get("type", "")

        if msg_type == "heartbeat_ack":
            self._last_heartbeat = time.time()
        elif msg_type == "status":
            self._esp32_status = msg
            self._last_heartbeat = time.time()
        elif msg_type == "error":
            logger.warning(f"ESP32报告错误: {msg}")
            self._error_count += 1

    def _queue_command(self, data: bytes) -> bool:
        """将命令加入发送队列"""
        try:
            self._send_queue.put_nowait(data)
            return True
        except queue.Full:
            # 丢弃最旧的，放入最新的
            try:
                _ = self._send_queue.get_nowait()
                self._send_queue.put_nowait(data)
                return True
            except queue.Empty:
                return False

    # ============ 公共API ============

    def send_mouse_move(self, dx: float, dy: float, buttons: int = 0) -> bool:
        """发送鼠标移动指令"""
        if self.cfg.dummy_esp32:
            self._dummy_accumulated_dx += dx
            self._dummy_accumulated_dy += dy
            if self.cfg.show_tx_rx:
                logger.debug(f"[DUMMY] Mouse move: dx={dx:.1f}, dy={dy:.1f}")
            return True

        if not self._connected:
            return False

        # 将浮点偏移转换为整数 (符合HID规范)
        dx_int = max(-127, min(127, int(round(dx))))
        dy_int = max(-127, min(127, int(round(dy))))

        if dx_int == 0 and dy_int == 0:
            return True

        data = self.encoder.encode_mouse_move(dx_int, dy_int, buttons)
        return self._queue_command(data)

    def send_mouse_click(self, button: int = 0) -> bool:
        """发送鼠标点击指令"""
        if self.cfg.dummy_esp32:
            logger.debug(f"[DUMMY] Mouse click: button={button}")
            return True

        if not self._connected:
            return False

        data = self.encoder.encode_mouse_button(button, True)
        success = self._queue_command(data)

        # 自动释放 (50ms后)
        if success:
            release_data = self.encoder.encode_mouse_button(button, False)
            # 立即入队，ESP32端会处理时序
            self._queue_command(release_data)

        return success

    def send_combined(self, dx: float, dy: float, buttons: int = 0,
                      wheel: int = 0) -> bool:
        """
        发送综合鼠标指令 (移动+按键+滚轮)
        效率更高，减少通信次数
        """
        if self.cfg.dummy_esp32:
            self._dummy_accumulated_dx += dx
            self._dummy_accumulated_dy += dy
            return True

        if not self._connected:
            return False

        dx_int = max(-127, min(127, int(round(dx))))
        dy_int = max(-127, min(127, int(round(dy))))

        data = self.encoder.encode_combined(dx_int, dy_int, buttons, wheel)
        return self._queue_command(data)

    def is_connected(self) -> bool:
        """是否已连接"""
        if self.cfg.dummy_esp32:
            return True
        return self._connected

    def get_status(self) -> Dict:
        """获取桥接状态"""
        return {
            "connected": self.is_connected(),
            "tx_count": self._tx_count,
            "rx_count": self._rx_count,
            "error_count": self._error_count,
            "queue_size": self._send_queue.qsize(),
            "esp32_status": self._esp32_status,
        }

    def get_stats(self) -> str:
        """获取统计信息字符串"""
        s = self.get_status()
        return (f"ESP32: {'Connected' if s['connected'] else 'Disconnected'} | "
                f"TX:{s['tx_count']} RX:{s['rx_count']} Err:{s['error_count']} "
                f"Queue:{s['queue_size']}")


# 简单的JSON协议 (用于调试)
class ESP32JsonBridge(ESP32Bridge):
    """使用JSON文本协议的ESP32桥接器 (用于调试)"""

    def send_mouse_move(self, dx: float, dy: float, buttons: int = 0) -> bool:
        """使用JSON格式发送"""
        if not self._connected and not self.cfg.dummy_esp32:
            return False

        import json
        msg = json.dumps({
            "cmd": "move",
            "dx": round(dx, 2),
            "dy": round(dy, 2),
            "btn": buttons,
            "ts": time.time()
        }) + '\n'

        if self.cfg.dummy_esp32:
            logger.debug(f"[JSON] {msg.strip()}")
            return True

        try:
            self._serial.write(msg.encode())
            return True
        except Exception as e:
            logger.warning(f"JSON发送失败: {e}")
            return False
