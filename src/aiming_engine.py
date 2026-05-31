"""
瞄准引擎：目标选择 + 平滑追踪 + 预测补偿
"""
import logging
import time
from typing import Tuple, List, Dict, Optional
from collections import deque
import numpy as np

from src.config import AIMING_CFG, MOUSE_CFG, COCO_KEYPOINTS
from src.rtmo_decoder import RTMODecoder

logger = logging.getLogger(__name__)


class SmoothTracker:
    """平滑追踪器：EMA 或 PID 控制"""

    def __init__(self, mode: str = "pid"):
        self.mode = mode

        # EMA 状态
        self.ema_x = 0.0
        self.ema_y = 0.0
        self.ema_initialized = False

        # PID 状态
        self.pid_integral_x = 0.0
        self.pid_integral_y = 0.0
        self.pid_prev_error_x = 0.0
        self.pid_prev_error_y = 0.0
        self.pid_last_time = time.time()

        # 移动历史 (用于预测)
        self.position_history = deque(maxlen=8)
        self.timestamp_history = deque(maxlen=8)

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        """
        更新追踪器并返回平滑后的目标位置
        Args:
            target_x, target_y: 当前帧检测到的目标位置 (屏幕坐标)
        Returns:
            smooth_x, smooth_y: 平滑后的瞄准位置
        """
        now = time.time()
        self.position_history.append((target_x, target_y))
        self.timestamp_history.append(now)

        if self.mode == "ema":
            return self._update_ema(target_x, target_y)
        else:
            return self._update_pid(target_x, target_y)

    def _update_ema(self, tx: float, ty: float) -> Tuple[float, float]:
        """指数移动平均"""
        alpha = AIMING_CFG.ema_alpha
        if not self.ema_initialized:
            self.ema_x = tx
            self.ema_y = ty
            self.ema_initialized = True
        else:
            self.ema_x = alpha * tx + (1 - alpha) * self.ema_x
            self.ema_y = alpha * ty + (1 - alpha) * self.ema_y
        return self.ema_x, self.ema_y

    def _update_pid(self, tx: float, ty: float) -> Tuple[float, float]:
        """PID 控制器"""
        now = time.time()
        dt = now - self.pid_last_time
        if dt <= 0:
            dt = 0.016  # 默认 60fps
        self.pid_last_time = now

        # 误差 (目标位置 - 当前平滑位置，但这里简化为直接控制偏移)
        # 实际上 PID 应该控制鼠标移动速度，这里简化为控制目标位置
        # 更好的做法：使用上一帧的平滑位置作为当前位置
        curr_x = self.ema_x if self.ema_initialized else tx
        curr_y = self.ema_y if self.ema_initialized else ty

        error_x = tx - curr_x
        error_y = ty - curr_y

        # 比例
        p_x = AIMING_CFG.pid_kp * error_x
        p_y = AIMING_CFG.pid_kp * error_y

        # 积分
        self.pid_integral_x += error_x * dt
        self.pid_integral_y += error_y * dt
        self.pid_integral_x = np.clip(self.pid_integral_x, 
                                       -AIMING_CFG.pid_integral_limit, 
                                       AIMING_CFG.pid_integral_limit)
        self.pid_integral_y = np.clip(self.pid_integral_y, 
                                       -AIMING_CFG.pid_integral_limit, 
                                       AIMING_CFG.pid_integral_limit)
        i_x = AIMING_CFG.pid_ki * self.pid_integral_x
        i_y = AIMING_CFG.pid_ki * self.pid_integral_y

        # 微分
        d_x = AIMING_CFG.pid_kd * (error_x - self.pid_prev_error_x) / dt
        d_y = AIMING_CFG.pid_kd * (error_y - self.pid_prev_error_y) / dt
        self.pid_prev_error_x = error_x
        self.pid_prev_error_y = error_y

        # 输出
        out_x = curr_x + p_x + i_x + d_x
        out_y = curr_y + p_y + i_y + d_y

        self.ema_x = out_x
        self.ema_y = out_y
        self.ema_initialized = True

        return out_x, out_y

    def predict_position(self, frames_ahead: int = 2) -> Tuple[float, float]:
        """
        基于历史位置预测未来位置 (线性速度预测)
        """
        if len(self.position_history) < 3:
            return self.ema_x, self.ema_y

        # 计算最近几帧的平均速度
        velocities = []
        for i in range(1, min(4, len(self.position_history))):
            p1 = self.position_history[-i]
            p0 = self.position_history[-(i+1)]
            t1 = self.timestamp_history[-i]
            t0 = self.timestamp_history[-(i+1)]
            dt = t1 - t0
            if dt > 0:
                vx = (p1[0] - p0[0]) / dt
                vy = (p1[1] - p0[1]) / dt
                velocities.append((vx, vy))

        if not velocities:
            return self.ema_x, self.ema_y

        avg_vx = np.mean([v[0] for v in velocities])
        avg_vy = np.mean([v[1] for v in velocities])

        # 预测
        dt_pred = 0.016 * frames_ahead  # 假设 60fps
        pred_x = self.ema_x + avg_vx * dt_pred
        pred_y = self.ema_y + avg_vy * dt_pred

        return pred_x, pred_y

    def reset(self):
        """重置追踪器"""
        self.ema_initialized = False
        self.pid_integral_x = 0.0
        self.pid_integral_y = 0.0
        self.pid_prev_error_x = 0.0
        self.pid_prev_error_y = 0.0
        self.position_history.clear()
        self.timestamp_history.clear()


class AimingEngine:
    """
    瞄准引擎主类
    - 目标选择策略
    - 瞄准点计算
    - 平滑追踪
    - 鼠标偏移计算
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.decoder = RTMODecoder()
        self.tracker = SmoothTracker(mode=AIMING_CFG.smooth_mode)

        # 当前锁定目标 (用于持续追踪同一目标)
        self.locked_target_id = None
        self.locked_target_bbox = None
        self.lock_frames = 0
        self.max_lock_frames = 5  # 丢失目标后保持锁定的帧数

        # 性能统计
        self.inference_times = deque(maxlen=30)

    def process(self, 
                persons: List[Dict], 
                current_mouse_x: float, 
                current_mouse_y: float) -> Optional[Tuple[float, float, Dict]]:
        """
        处理一帧检测结果，返回鼠标移动偏移
        Args:
            persons: RTMO 解码后的人体列表
            current_mouse_x, current_mouse_y: 当前鼠标位置 (屏幕坐标)
        Returns:
            (dx, dy, target_person) 或 None (无目标)
        """
        if not persons:
            self.lock_frames += 1
            if self.lock_frames >= self.max_lock_frames:
                self.locked_target_id = None
                self.tracker.reset()
            return None

        # 1. 目标选择
        target = self._select_target(persons, current_mouse_x, current_mouse_y)
        if target is None:
            return None

        # 2. 获取瞄准点
        aim_x, aim_y, aim_conf = self.decoder.get_aim_point(target)
        if aim_conf < 0.1:
            return None

        # 3. 预测补偿
        if AIMING_CFG.enable_prediction:
            pred_x, pred_y = self.tracker.predict_position(AIMING_CFG.prediction_frames)
            # 混合预测和检测 (信任检测更多)
            blend = 0.7
            aim_x = blend * aim_x + (1 - blend) * pred_x
            aim_y = blend * aim_y + (1 - blend) * pred_y

        # 4. 平滑追踪
        smooth_x, smooth_y = self.tracker.update(aim_x, aim_y)

        # 5. 计算鼠标偏移 (从当前位置到目标位置的差值)
        # 注意：这里假设输入图像是全屏画面，鼠标位置与画面坐标对应
        dx = smooth_x - current_mouse_x
        dy = smooth_y - current_mouse_y

        # 6. 应用灵敏度
        dx *= MOUSE_CFG.sensitivity_x
        dy *= MOUSE_CFG.sensitivity_y

        # 7. 速度限制
        max_move = AIMING_CFG.max_move_per_frame
        dist = np.sqrt(dx**2 + dy**2)
        if dist > max_move:
            scale = max_move / dist
            dx *= scale
            dy *= scale

        # 8. 最小阈值过滤
        if abs(dx) < AIMING_CFG.min_move_threshold:
            dx = 0
        if abs(dy) < AIMING_CFG.min_move_threshold:
            dy = 0

        return (dx, dy, target)

    def _select_target(self, 
                       persons: List[Dict], 
                       mouse_x: float, 
                       mouse_y: float) -> Optional[Dict]:
        """
        根据策略选择目标
        """
        strategy = AIMING_CFG.target_select_strategy
        screen_cx = self.screen_w / 2
        screen_cy = self.screen_h / 2

        # 首先过滤不在瞄准区域内的目标
        valid_persons = []
        for p in persons:
            cx = p["center"][0]
            cy = p["center"][1]
            rx1 = AIMING_CFG.aim_region_x[0] * self.screen_w
            rx2 = AIMING_CFG.aim_region_x[1] * self.screen_w
            ry1 = AIMING_CFG.aim_region_y[0] * self.screen_h
            ry2 = AIMING_CFG.aim_region_y[1] * self.screen_h
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                valid_persons.append(p)

        if not valid_persons:
            valid_persons = persons  # 如果没有在区域内的，放宽限制

        # 尝试保持锁定同一目标
        if self.locked_target_id is not None:
            best_match = None
            best_iou = 0.3
            for p in valid_persons:
                iou = self._compute_iou(self.locked_target_bbox, p["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_match = p
            if best_match is not None:
                self.locked_target_bbox = best_match["bbox"]
                self.lock_frames = 0
                return best_match

        # 根据策略选择新目标
        if strategy == "nearest":
            # 选择距离屏幕中心最近的目标
            best = None
            best_dist = float('inf')
            for p in valid_persons:
                dist = np.sqrt((p["center"][0] - screen_cx)**2 + 
                               (p["center"][1] - screen_cy)**2)
                if dist < best_dist:
                    best_dist = dist
                    best = p
            target = best

        elif strategy == "center":
            # 选择最接近准星的目标
            best = None
            best_dist = float('inf')
            for p in valid_persons:
                dist = np.sqrt((p["center"][0] - mouse_x)**2 + 
                               (p["center"][1] - mouse_y)**2)
                if dist < best_dist:
                    best_dist = dist
                    best = p
            target = best

        elif strategy == "largest":
            # 选择最大的目标 (最近距离)
            target = max(valid_persons, key=lambda p: p["height"])

        elif strategy == "highest_conf":
            # 选择置信度最高的
            target = max(valid_persons, key=lambda p: p["score"])

        else:
            target = valid_persons[0] if valid_persons else None

        # 更新锁定状态
        if target is not None:
            self.locked_target_id = id(target)
            self.locked_target_bbox = target["bbox"]
            self.lock_frames = 0

        return target

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个 bbox 的 IoU"""
        if box1 is None or box2 is None:
            return 0.0
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
