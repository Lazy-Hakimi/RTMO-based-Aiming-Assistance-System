"""
瞄准引擎增强版 (Aiming Engine Enhanced)
新增功能:
1. 身体朝向感知的多目标优先级选择
2. 压枪补偿集成
3. 准星校准集成

保持原有功能不变:
- PID平滑追踪
- 预测补偿
- EMA平滑
"""
import logging
import time
from typing import Tuple, List, Dict, Optional
from collections import deque
import numpy as np

from src.config import (
    AIMING_CFG, MOUSE_CFG, BODY_ORI_CFG, TARGET_PRIO_CFG,
    RECOIL_CFG, CALIB_CFG, COCO_KEYPOINTS
)
from src.rtmo_decoder import RTMODecoder
from src.body_orientation import BodyOrientationEstimator, compute_orientation_score
from src.recoil_compensator import RecoilCompensator
from src.crosshair_calibrator import CrosshairCalibrator

logger = logging.getLogger(__name__)


class SmoothTracker:
    """平滑追踪器：EMA 或 PID 控制 (保留原版)"""

    def __init__(self, mode: str = "pid"):
        self.mode = mode
        self.ema_x = 0.0
        self.ema_y = 0.0
        self.ema_initialized = False
        self.pid_integral_x = 0.0
        self.pid_integral_y = 0.0
        self.pid_prev_error_x = 0.0
        self.pid_prev_error_y = 0.0
        self.pid_last_time = time.time()
        self.position_history = deque(maxlen=8)
        self.timestamp_history = deque(maxlen=8)

    def update(self, target_x: float, target_y: float) -> Tuple[float, float]:
        now = time.time()
        self.position_history.append((target_x, target_y))
        self.timestamp_history.append(now)

        if self.mode == "ema":
            return self._update_ema(target_x, target_y)
        else:
            return self._update_pid(target_x, target_y)

    def _update_ema(self, tx: float, ty: float) -> Tuple[float, float]:
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
        now = time.time()
        dt = now - self.pid_last_time
        if dt <= 0:
            dt = 0.016
        self.pid_last_time = now

        curr_x = self.ema_x if self.ema_initialized else tx
        curr_y = self.ema_y if self.ema_initialized else ty

        error_x = tx - curr_x
        error_y = ty - curr_y

        p_x = AIMING_CFG.pid_kp * error_x
        p_y = AIMING_CFG.pid_kp * error_y

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

        d_x = AIMING_CFG.pid_kd * (error_x - self.pid_prev_error_x) / dt
        d_y = AIMING_CFG.pid_kd * (error_y - self.pid_prev_error_y) / dt
        self.pid_prev_error_x = error_x
        self.pid_prev_error_y = error_y

        out_x = curr_x + p_x + i_x + d_x
        out_y = curr_y + p_y + i_y + d_y

        self.ema_x = out_x
        self.ema_y = out_y
        self.ema_initialized = True

        return out_x, out_y

    def predict_position(self, frames_ahead: int = 2) -> Tuple[float, float]:
        if len(self.position_history) < 3:
            return self.ema_x, self.ema_y

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

        dt_pred = 0.016 * frames_ahead
        pred_x = self.ema_x + avg_vx * dt_pred
        pred_y = self.ema_y + avg_vy * dt_pred

        return pred_x, pred_y

    def reset(self):
        self.ema_initialized = False
        self.pid_integral_x = 0.0
        self.pid_integral_y = 0.0
        self.pid_prev_error_x = 0.0
        self.pid_prev_error_y = 0.0
        self.position_history.clear()
        self.timestamp_history.clear()


class TargetPrioritizer:
    """
    多目标优先级选择器
    基于身体朝向、距离、威胁度等综合权重选择最优目标
    """

    def __init__(self):
        self.cfg = TARGET_PRIO_CFG
        self.ori_estimator = BodyOrientationEstimator()

    def select_target(self,
                      persons: List[Dict],
                      mouse_x: float,
                      mouse_y: float,
                      screen_w: int,
                      screen_h: int) -> Optional[Dict]:
        """
        从多个人体中选择最优目标
        
        Returns:
            最优目标person字典，带有额外的"priority_info"字段
        """
        if not persons:
            return None

        # 1. 筛选在瞄准区域内的目标
        valid_persons = self._filter_by_aim_region(persons, screen_w, screen_h)
        if not valid_persons:
            valid_persons = persons

        # 2. 计算每个人的优先级分数
        scored_persons = []
        for person in valid_persons:
            score, info = self._compute_priority_score(
                person, mouse_x, mouse_y, screen_w, screen_h
            )
            person["priority_score"] = score
            person["priority_info"] = info
            scored_persons.append((score, person))

        # 3. 按分数排序
        scored_persons.sort(key=lambda x: x[0], reverse=True)

        if not scored_persons:
            return None

        return scored_persons[0][1]

    def _filter_by_aim_region(self, persons: List[Dict],
                               screen_w: int, screen_h: int) -> List[Dict]:
        """筛选在瞄准区域内的目标"""
        rx1 = int(TARGET_PRIO_CFG.aim_region_x[0] * screen_w)
        rx2 = int(TARGET_PRIO_CFG.aim_region_x[1] * screen_w)
        ry1 = int(TARGET_PRIO_CFG.aim_region_y[0] * screen_h)
        ry2 = int(TARGET_PRIO_CFG.aim_region_y[1] * screen_h)

        valid = []
        for p in persons:
            cx, cy = p["center"]
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                valid.append(p)

        return valid

    def _compute_priority_score(self, person: Dict,
                                 mouse_x: float, mouse_y: float,
                                 screen_w: int, screen_h: int) -> Tuple[float, Dict]:
        """
        计算单个目标的优先级分数
        
        综合权重公式:
        Score = w_dist * score_dist + w_threat * score_threat + 
                w_ori * score_ori + w_size * score_size + w_conf * score_conf
        """
        cfg = self.cfg
        info = {}

        # 1. 距离分数 (离准星越近分数越高，使用高斯衰减)
        target_cx, target_cy = person["center"]
        dist = np.sqrt((target_cx - mouse_x)**2 + (target_cy - mouse_y)**2)

        if cfg.distance_decay == "gaussian":
            sigma = cfg.max_aim_distance / 3.0
            score_dist = np.exp(-(dist**2) / (2 * sigma**2))
        else:
            score_dist = max(0, 1.0 - dist / cfg.max_aim_distance)

        info["distance"] = dist
        info["score_dist"] = score_dist

        # 2. 朝向分数 (身体朝向)
        score_ori = 0.5
        facing_info = "unknown"

        if cfg.use_orientation and BODY_ORI_CFG.enabled:
            orientation = self.ori_estimator.estimate(person)
            person["body_orientation"] = orientation
            facing = orientation.get("body_facing", "unknown")
            facing_info = facing

            if facing == "front":
                # 正面朝向 - 威胁最高，优先击杀
                score_ori = 1.0
            elif facing in ("left", "right"):
                # 侧面 - 容易击杀
                if cfg.prioritize_back:
                    score_ori = 0.8
                else:
                    score_ori = 0.5
            elif facing == "back":
                # 背面
                if cfg.prioritize_back:
                    score_ori = 0.9  # 背身容易击杀
                else:
                    score_ori = 0.2
            else:
                score_ori = 0.5

            info["body_facing"] = facing
            info["facing_confidence"] = orientation.get("confidence", 0)

        info["score_ori"] = score_ori

        # 3. 威胁度分数 (基于朝向和距离的综合)
        # 正面且距离近的威胁最高
        if facing_info == "front":
            score_threat = score_dist * 1.0
        elif facing_info == "back":
            score_threat = score_dist * 0.3
        else:
            score_threat = score_dist * 0.6

        info["score_threat"] = score_threat

        # 4. 大小分数 (越大越明显)
        person_height = person.get("height", 0)
        max_h = screen_h * 0.8
        score_size = min(1.0, person_height / max_h) if max_h > 0 else 0.5
        info["score_size"] = score_size
        info["person_height"] = person_height

        # 5. 置信度分数
        score_conf = person.get("score", 0.5)
        info["score_conf"] = score_conf

        # 综合分数
        total_score = (
            cfg.w_distance * score_dist +
            cfg.w_threat * score_threat +
            cfg.w_orientation * score_ori +
            cfg.w_size * score_size +
            cfg.w_confidence * score_conf
        )

        info["total_score"] = total_score
        return total_score, info

    def reset(self):
        """重置选择器状态"""
        self.ori_estimator.reset()


class AimingEngine:
    """
    瞄准引擎增强版
    集成: 目标优先级选择 + 身体朝向 + 压枪补偿 + 准星校准
    """

    def __init__(self, screen_width: int = 1920, screen_height: int = 1080):
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.decoder = RTMODecoder()
        self.tracker = SmoothTracker(mode=AIMING_CFG.smooth_mode)

        # 新增模块
        self.target_prioritizer = TargetPrioritizer()
        self.recoil_compensator = RecoilCompensator()
        self.calibrator = CrosshairCalibrator()

        # 锁定状态
        self.locked_target_id = None
        self.locked_target_bbox = None
        self.lock_frames = 0
        self.max_lock_frames = TARGET_PRIO_CFG.lock_keep_frames

        # 性能统计
        self.inference_times = deque(maxlen=30)

        # 开火状态
        self._was_firing = False

    def process(self,
                persons: List[Dict],
                current_mouse_x: float,
                current_mouse_y: float,
                should_fire: bool = False) -> Optional[Tuple[float, float, Dict]]:
        """
        处理一帧检测结果，返回鼠标移动偏移
        
        Args:
            persons: RTMO解码后的人体列表
            current_mouse_x, current_mouse_y: 当前鼠标位置
            should_fire: 是否开火信号
            
        Returns:
            (dx, dy, target_person) 或 None
        """
        # ===== 1. 处理开火状态 =====
        if should_fire and not self._was_firing:
            self.recoil_compensator.on_fire_start()
            self._was_firing = True
        elif not should_fire and self._was_firing:
            self.recoil_compensator.on_fire_stop()
            self._was_firing = False

        if should_fire:
            self.recoil_compensator.on_shot_fired(hit_target=True)

        # ===== 2. 目标选择 =====
        if not persons:
            self.lock_frames += 1
            if self.lock_frames >= self.max_lock_frames:
                self.locked_target_id = None
                self.tracker.reset()
                self.target_prioritizer.reset()
            return None

        # 使用优先级选择器
        target = self.target_prioritizer.select_target(
            persons, current_mouse_x, current_mouse_y,
            self.screen_w, self.screen_h
        )

        if target is None:
            return None

        # ===== 3. 获取瞄准点 =====
        aim_x, aim_y, aim_conf = self.decoder.get_aim_point(target)
        if aim_conf < 0.1:
            return None

        # ===== 4. 应用准星校准偏移 =====
        if AIMING_CFG.enable_calibration and self.calibrator.cfg.enabled:
            target_dx = aim_x - current_mouse_x
            target_dy = aim_y - current_mouse_y
            aim_x, aim_y = self.calibrator.apply_calibration(aim_x, aim_y, target_dx, target_dy)

        # ===== 5. 预测补偿 =====
        if AIMING_CFG.enable_prediction:
            pred_x, pred_y = self.tracker.predict_position(AIMING_CFG.prediction_frames)
            blend = 0.7
            aim_x = blend * aim_x + (1 - blend) * pred_x
            aim_y = blend * aim_y + (1 - blend) * pred_y

        # ===== 6. 平滑追踪 =====
        smooth_x, smooth_y = self.tracker.update(aim_x, aim_y)

        # ===== 7. 计算鼠标偏移 =====
        dx = smooth_x - current_mouse_x
        dy = smooth_y - current_mouse_y

        # ===== 8. 应用灵敏度 =====
        dx *= MOUSE_CFG.sensitivity_x
        dy *= MOUSE_CFG.sensitivity_y

        # ===== 9. 应用压枪补偿 =====
        if AIMING_CFG.enable_recoil_compensation and RECOIL_CFG.enabled:
            recoil_x, recoil_y = self.recoil_compensator.get_compensation_offset(dx, dy)
            dx += recoil_x
            dy += recoil_y

        # ===== 10. 速度限制 =====
        max_move = AIMING_CFG.max_move_per_frame
        dist = np.sqrt(dx**2 + dy**2)
        if dist > max_move:
            scale = max_move / dist
            dx *= scale
            dy *= scale

        # ===== 11. 最小阈值过滤 =====
        if abs(dx) < AIMING_CFG.min_move_threshold:
            dx = 0
        if abs(dy) < AIMING_CFG.min_move_threshold:
            dy = 0

        # 更新锁定状态
        self.locked_target_id = id(target)
        self.locked_target_bbox = target.get("bbox")
        self.lock_frames = 0

        return (dx, dy, target)

    def set_weapon(self, weapon_name: str):
        """切换武器类型"""
        self.recoil_compensator.set_weapon(weapon_name)

    def get_recoil_status(self) -> Dict:
        """获取压枪状态"""
        return self.recoil_compensator.get_status()

    def get_calibration_info(self) -> Dict:
        """获取校准信息"""
        return self.calibrator.get_debug_info()

    def reset(self):
        """重置引擎状态"""
        self.locked_target_id = None
        self.locked_target_bbox = None
        self.lock_frames = 0
        self.tracker.reset()
        self.target_prioritizer.reset()
        self.recoil_compensator.reset()
        self._was_firing = False

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
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
