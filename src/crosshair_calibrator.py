"""
准星校准系统 (Crosshair Calibrator)
支持自动和手动两种模式校准准星偏移

背景说明:
- 在FPS游戏中，鼠标准星通常固定在画面正中心
- 移动鼠标会直接移动画面（改变视角）
- 姿态估计模型给出的瞄准点（如头部关键点）与准星之间存在系统偏移
- 本系统用于测量和补偿这种偏移

校准模式:
1. 手动校准: 用户通过键盘快捷键微调瞄准点偏移
2. 自动校准: 使用模板匹配自动检测准星位置并计算偏移
3. 半自动校准: 自动检测+手动确认

校准参数:
- aim_offset_x/y: 瞄准点基础偏移 (像素)
- sensitivity_multiplier: 灵敏度乘数 (用于微调)
- calibration_matrix: 8方向校准矩阵 (更精细的校准)
"""
import os
import json
import logging
import time
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

from src.config import CALIB_CFG, AIMING_CFG, CAPTURE_CFG

logger = logging.getLogger(__name__)


@dataclass
class CalibrationData:
    """校准数据"""
    offset_x: float = 0.0           # 水平偏移
    offset_y: float = 0.0           # 垂直偏移
    sensitivity_x: float = 1.0      # X灵敏度
    sensitivity_y: float = 1.0      # Y灵敏度
    # 8方向校准矩阵 (上, 右上, 右, 右下, 下, 左下, 左, 左上)
    directional_multipliers: List[float] = None
    timestamp: str = ""

    def __post_init__(self):
        if self.directional_multipliers is None:
            self.directional_multipliers = [1.0] * 8
        if not self.timestamp:
            self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")


class CrosshairCalibrator:
    """
    准星校准器
    提供多种校准方式来确定准确的瞄准偏移
    """

    # 8方向角度定义 (度)
    DIRECTIONS = [
        ("up", 0, -1),        # 上
        ("up_right", 1, -1),  # 右上
        ("right", 1, 0),      # 右
        ("down_right", 1, 1), # 右下
        ("down", 0, 1),       # 下
        ("down_left", -1, 1), # 左下
        ("left", -1, 0),      # 左
        ("up_left", -1, -1),  # 左上
    ]

    # OpenCV方向键映射
    KEY_ADJUSTMENTS = {
        ord('w'): (0.0, -1.0),   # 上
        ord('s'): (0.0, 1.0),    # 下
        ord('a'): (-1.0, 0.0),   # 左
        ord('d'): (1.0, 0.0),    # 右
        ord('W'): (0.0, -5.0),   # 上 (大步)
        ord('S'): (0.0, 5.0),    # 下 (大步)
        ord('A'): (-5.0, 0.0),   # 左 (大步)
        ord('D'): (5.0, 0.0),    # 右 (大步)
    }

    def __init__(self):
        self.cfg = CALIB_CFG
        self.data = CalibrationData()

        # 加载已有校准数据
        self._load_calibration()

        # 校准状态
        self.is_calibrating = False
        self.current_step = 0
        self.calibration_results = []

        # 准星模板 (用于自动检测)
        self.crosshair_template = None

    def _load_calibration(self):
        """从文件加载校准数据"""
        if os.path.exists(self.cfg.calibration_file):
            try:
                with open(self.cfg.calibration_file, 'r') as f:
                    saved = json.load(f)
                self.data.offset_x = saved.get("offset_x", 0.0)
                self.data.offset_y = saved.get("offset_y", 0.0)
                self.data.sensitivity_x = saved.get("sensitivity_x", 1.0)
                self.data.sensitivity_y = saved.get("sensitivity_y", 1.0)
                self.data.directional_multipliers = saved.get(
                    "directional_multipliers", [1.0] * 8
                )
                logger.info(f"已加载校准数据: offset=({self.data.offset_x:.1f}, "
                           f"{self.data.offset_y:.1f})")
            except Exception as e:
                logger.warning(f"加载校准数据失败: {e}")

    def save_calibration(self):
        """保存校准数据到文件"""
        try:
            os.makedirs(os.path.dirname(self.cfg.calibration_file), exist_ok=True)
            self.data.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.cfg.calibration_file, 'w') as f:
                json.dump(asdict(self.data), f, indent=2)
            logger.info(f"校准数据已保存: {self.cfg.calibration_file}")
            return True
        except Exception as e:
            logger.error(f"保存校准数据失败: {e}")
            return False

    # ============ 手动校准 ============

    def start_manual_calibration(self):
        """开始手动校准模式"""
        self.is_calibrating = True
        self.current_step = 0
        logger.info("=" * 50)
        logger.info("手动准星校准模式")
        logger.info("=" * 50)
        logger.info("操作说明:")
        logger.info("  W/A/S/D - 微调瞄准点偏移 (1像素)")
        logger.info("  Shift+W/A/S/D - 粗调瞄准点偏移 (5像素)")
        logger.info("  +/- - 调整灵敏度")
        logger.info("  Enter - 保存并退出")
        logger.info("  R - 重置偏移")
        logger.info("  ESC - 取消校准")
        logger.info(f"  当前偏移: ({self.data.offset_x:.1f}, {self.data.offset_y:.1f})")

    def handle_key(self, key: int) -> bool:
        """
        处理校准模式下的按键
        
        Returns:
            False: 退出校准模式
            True: 继续校准
        """
        if key == 27:  # ESC
            self.is_calibrating = False
            logger.info("校准已取消")
            return False

        elif key == 13:  # Enter
            self.save_calibration()
            self.is_calibrating = False
            logger.info("校准完成并保存")
            # 同步更新AIMING_CFG
            AIMING_CFG.aim_offset_x = self.data.offset_x
            AIMING_CFG.aim_offset_y = self.data.offset_y
            return False

        elif key == ord('r') or key == ord('R'):
            self.data.offset_x = 0.0
            self.data.offset_y = 0.0
            logger.info("偏移已重置")

        elif key == ord('+') or key == ord('='):
            self.data.sensitivity_x = min(3.0, self.data.sensitivity_x + 0.1)
            self.data.sensitivity_y = min(3.0, self.data.sensitivity_y + 0.1)
            logger.info(f"灵敏度增加到: {self.data.sensitivity_x:.2f}")

        elif key == ord('-') or key == ord('_'):
            self.data.sensitivity_x = max(0.1, self.data.sensitivity_x - 0.1)
            self.data.sensitivity_y = max(0.1, self.data.sensitivity_y - 0.1)
            logger.info(f"灵敏度降低到: {self.data.sensitivity_x:.2f}")

        elif key in self.KEY_ADJUSTMENTS:
            dx, dy = self.KEY_ADJUSTMENTS[key]
            self.data.offset_x += dx
            self.data.offset_y += dy
            logger.info(f"偏移调整: ({self.data.offset_x:.1f}, {self.data.offset_y:.1f})")

        return True

    # ============ 自动校准 ============

    def auto_calibrate(self, frame: np.ndarray) -> bool:
        """
        自动校准 - 检测画面中的准星位置
        
        原理:
        1. 检测准星标记 (十字线、T型准星等)
        2. 计算准星中心与画面中心的偏移
        3. 将偏移量设为校准参数
        
        Args:
            frame: 当前画面帧
            
        Returns:
            True: 校准成功
            False: 校准失败
        """
        if frame is None:
            return False

        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2

        # 方法1: 在中心区域检测准星 (通过颜色和形状)
        roi_size = 100
        x1, y1 = max(0, cx - roi_size), max(0, cy - roi_size)
        x2, y2 = min(w, cx + roi_size), min(h, cy + roi_size)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return False

        # 尝试检测准星 (使用颜色阈值和形态学操作)
        detected, crosshair_pos = self._detect_crosshair_in_roi(roi)

        if detected:
            # 计算相对于画面中心的偏移
            global_cx = x1 + crosshair_pos[0]
            global_cy = y1 + crosshair_pos[1]

            offset_x = global_cx - cx
            offset_y = global_cy - cy

            self.data.offset_x = offset_x
            self.data.offset_y = offset_y

            logger.info(f"自动校准成功: 准星偏移=({offset_x:.1f}, {offset_y:.1f})")
            self.save_calibration()

            # 同步更新AIMING_CFG
            AIMING_CFG.aim_offset_x = offset_x
            AIMING_CFG.aim_offset_y = offset_y
            return True
        else:
            logger.warning("自动校准失败: 未检测到准星")
            return False

    def _detect_crosshair_in_roi(self, roi: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """
        在ROI区域中检测准星
        
        检测方法:
        1. 转换为灰度
        2. 边缘检测
        3. 查找十字交叉线
        4. 计算交叉点
        
        Returns:
            (detected, (x, y)): 是否检测到，以及准星位置(ROI坐标)
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 方法1: 边缘检测 + 霍夫线变换
        edges = cv2.Canny(gray, 50, 150)

        # 查找直线
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20,
                                minLineLength=10, maxLineGap=3)

        if lines is None or len(lines) < 2:
            # 方法2: 通过亮度中心检测
            return self._detect_by_brightness_center(gray)

        # 分类水平和垂直线
        h_lines = []  # 水平线
        v_lines = []  # 垂直线

        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)

            if dx > dy * 2:  # 水平线
                h_lines.append(line[0])
            elif dy > dx * 2:  # 垂直线
                v_lines.append(line[0])

        # 查找十字交叉
        if len(h_lines) > 0 and len(v_lines) > 0:
            # 计算水平线中心
            h_centers = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in h_lines]
            v_centers = [((x1+x2)//2, (y1+y2)//2) for x1, y1, x2, y2 in v_lines]

            # 找最接近的交叉点
            best_dist = float('inf')
            best_cross = (roi.shape[1]//2, roi.shape[0]//2)

            for hx, hy in h_centers:
                for vx, vy in v_centers:
                    dist = abs(hx - vx) + abs(hy - vy)
                    if dist < best_dist:
                        best_dist = dist
                        best_cross = ((hx + vx) // 2, (hy + vy) // 2)

            if best_dist < 20:  # 交叉点距离阈值
                return True, best_cross

        return False, (0, 0)

    def _detect_by_brightness_center(self, gray: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """通过亮度中心检测准星 (适用于发光准星)"""
        # 查找亮中心 (准星通常比周围更亮)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 查找轮廓
        contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 找最接近中心的轮廓
            h, w = gray.shape
            center = (w // 2, h // 2)

            best_dist = float('inf')
            best_pos = center

            for cnt in contours:
                M = cv2.moments(cnt)
                if M["m00"] > 5:  # 足够大的亮斑
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dist = abs(cx - center[0]) + abs(cy - center[1])
                    if dist < best_dist:
                        best_dist = dist
                        best_pos = (cx, cy)

            if best_dist < 50:
                return True, best_pos

        return False, (0, 0)

    # ============ 校准数据应用 ============

    def apply_calibration(self, aim_x: float, aim_y: float,
                         target_dx: float, target_dy: float) -> Tuple[float, float]:
        """
        应用校准偏移到瞄准点
        
        Args:
            aim_x, aim_y: 原始瞄准点坐标
            target_dx, target_dy: 从画面中心到目标的偏移方向
            
        Returns:
            calibrated_x, calibrated_y: 校准后的瞄准点
        """
        # 基础偏移
        calibrated_x = aim_x + self.data.offset_x
        calibrated_y = aim_y + self.data.offset_y

        # 8方向校准
        if self.data.directional_multipliers and len(self.data.directional_multipliers) == 8:
            # 计算目标方向索引
            angle = np.degrees(np.arctan2(target_dy, target_dx))
            angle = (angle + 360) % 360
            dir_idx = int(angle / 45) % 8

            multiplier = self.data.directional_multipliers[dir_idx]
            calibrated_x += target_dx * (multiplier - 1.0) * 0.1
            calibrated_y += target_dy * (multiplier - 1.0) * 0.1

        return calibrated_x, calibrated_y

    def get_calibration_offset(self) -> Tuple[float, float]:
        """获取当前校准偏移量"""
        return self.data.offset_x, self.data.offset_y

    def get_sensitivity(self) -> Tuple[float, float]:
        """获取当前灵敏度设置"""
        return self.data.sensitivity_x, self.data.sensitivity_y

    def is_active(self) -> bool:
        """是否在校准模式"""
        return self.is_calibrating

    def get_debug_info(self) -> Dict:
        """获取调试信息"""
        return {
            "offset_x": self.data.offset_x,
            "offset_y": self.data.offset_y,
            "sensitivity_x": self.data.sensitivity_x,
            "sensitivity_y": self.data.sensitivity_y,
            "is_calibrating": self.is_calibrating,
            "directional_multipliers": self.data.directional_multipliers,
        }

    def draw_calibration_ui(self, frame: np.ndarray) -> np.ndarray:
        """在画面上绘制校准UI"""
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 半透明黑色背景
        overlay = np.zeros_like(vis)
        cv2.rectangle(overlay, (10, 10), (350, 160), (0, 0, 0), -1)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.7, 0)

        # 标题
        cv2.putText(vis, "=== CROSSHAIR CALIBRATION ===", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        # 偏移信息
        cv2.putText(vis, f"Offset: ({self.data.offset_x:+.1f}, {self.data.offset_y:+.1f})",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(vis, f"Sensitivity: {self.data.sensitivity_x:.2f}",
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 操作提示
        cv2.putText(vis, "W/A/S/D: Adjust  +/-: Sens  Enter: Save",
                   (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(vis, "R: Reset  ESC: Cancel",
                   (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # 绘制当前偏移指示
        cx, cy = w // 2, h // 2
        offset_cx = int(cx + self.data.offset_x * 2)  # 放大显示
        offset_cy = int(cy + self.data.offset_y * 2)
        cv2.circle(vis, (offset_cx, offset_cy), 8, (0, 0, 255), 2)
        cv2.line(vis, (cx, cy), (offset_cx, offset_cy), (0, 0, 255), 1)

        return vis
