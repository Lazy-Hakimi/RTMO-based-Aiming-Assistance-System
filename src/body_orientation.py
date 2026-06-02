"""
身体/枪口朝向估算模块
基于RTMO输出的2D人体关键点，估算身体朝向和枪口方向

核心算法:
1. 身体朝向 - 通过肩线和髋线的向量分析，判断人体正面/侧面/背面
2. 枪口朝向 - 通过手臂关键点姿态，估算武器指向方向
3. 朝向平滑 - 使用历史帧进行时间域平滑，减少抖动

学术参考:
- 肩线向量法: 使用左右肩关键点构建水平参考向量
- 髋线向量法: 使用左右髋关键点辅助验证
- 躯干角度法: 肩中点到髋中点的向量与垂直方向的夹角
"""
import logging
import math
from typing import Dict, Tuple, List, Optional
from collections import deque
import numpy as np

from src.config import BODY_ORI_CFG, COCO_KEYPOINTS

logger = logging.getLogger(__name__)


class BodyOrientationEstimator:
    """
    身体朝向估算器
    利用COCO格式的17个关键点中的肩部和髋部关键点估算人体朝向
    """

    def __init__(self):
        self.cfg = BODY_ORI_CFG

        # 历史记录 (用于平滑)
        self._ori_history: deque = deque(maxlen=self.cfg.history_smooth_frames)
        self._muzzle_history: deque = deque(maxlen=self.cfg.history_smooth_frames)

    def estimate(self, person: Dict) -> Dict:
        """
        估算单个人体的朝向信息
        
        Args:
            person: RTMO解码后的人体字典，包含 "keypoints" [17, 3]
            
        Returns:
            orientation_info: 包含以下字段的字典:
                - body_facing: str - "front"|"left"|"right"|"back"|"unknown"
                - body_angle: float - 身体朝向角度 (0=正面, 90=侧面, 180=背面)
                - muzzle_dir: str - 枪口朝向 "forward"|"left"|"right"|"up"|"down"|"unknown"
                - muzzle_angle: float - 枪口角度 (度)
                - shoulder_width: float - 肩宽 (像素)
                - hip_width: float - 髋宽 (像素)
                - torso_tilt: float - 躯干倾斜角 (度)
                - confidence: float - 朝向估算置信度 0-1
        """
        keypoints = person.get("keypoints", None)
        if keypoints is None or len(keypoints) < 17:
            return self._default_orientation()

        cfg = self.cfg

        # 提取关键点坐标和可见性
        ls = keypoints[cfg.shoulder_left_idx]   # left_shoulder
        rs = keypoints[cfg.shoulder_right_idx]  # right_shoulder
        lh = keypoints[cfg.hip_left_idx]        # left_hip
        rh = keypoints[cfg.hip_right_idx]       # right_hip
        nose = keypoints[cfg.nose_idx]           # nose

        # 检查关键点可见性
        ls_vis = ls[2] > cfg.kpt_visible_thresh
        rs_vis = rs[2] > cfg.kpt_visible_thresh
        lh_vis = lh[2] > cfg.kpt_visible_thresh
        rh_vis = rh[2] > cfg.kpt_visible_thresh
        nose_vis = nose[2] > cfg.kpt_visible_thresh

        # 至少需要一侧肩和髋可见
        if not (ls_vis or rs_vis) or not (lh_vis or rh_vis):
            return self._default_orientation()

        # ========== 1. 计算肩宽和髋宽 ==========
        shoulder_width = 0.0
        hip_width = 0.0

        if ls_vis and rs_vis:
            shoulder_width = float(np.linalg.norm(rs[:2] - ls[:2]))
        if lh_vis and rh_vis:
            hip_width = float(np.linalg.norm(rh[:2] - lh[:2]))

        # ========== 2. 计算躯干倾斜角 ==========
        # 肩中点和髋中点
        shoulder_mid = np.array([0.0, 0.0])
        hip_mid = np.array([0.0, 0.0])

        valid_shoulder = 0
        if ls_vis:
            shoulder_mid += ls[:2]
            valid_shoulder += 1
        if rs_vis:
            shoulder_mid += rs[:2]
            valid_shoulder += 1
        if valid_shoulder > 0:
            shoulder_mid /= valid_shoulder

        valid_hip = 0
        if lh_vis:
            hip_mid += lh[:2]
            valid_hip += 1
        if rh_vis:
            hip_mid += rh[:2]
            valid_hip += 1
        if valid_hip > 0:
            hip_mid /= valid_hip

        # 躯干向量 (从髋到肩)
        torso_vector = shoulder_mid - hip_mid
        torso_length = np.linalg.norm(torso_vector)

        # 躯干与垂直方向的夹角
        vertical = np.array([0.0, -1.0])  # 图像坐标系，y轴向下为正，所以垂直向上是[0,-1]
        if torso_length > 1e-6:
            cos_tilt = np.dot(torso_vector, vertical) / torso_length
            cos_tilt = np.clip(cos_tilt, -1.0, 1.0)
            torso_tilt = math.degrees(math.acos(cos_tilt))
        else:
            torso_tilt = 0.0

        # ========== 3. 身体朝向判断 (核心算法) ==========
        # 方法: 肩线向量法 + 宽高比法 + 鼻子位置验证

        body_facing = "unknown"
        body_angle = 90.0  # 默认侧面
        confidence = 0.0

        if ls_vis and rs_vis and lh_vis and rh_vis:
            # 所有关键点可见，使用综合判断

            # 肩线向量 (左肩 -> 右肩)
            shoulder_vec = rs[:2] - ls[:2]
            shoulder_angle = math.degrees(math.atan2(abs(shoulder_vec[1]), abs(shoulder_vec[0])))

            # 髋线向量
            hip_vec = rh[:2] - lh[:2]

            # 肩宽/髋宽比
            if hip_width > 0:
                wh_ratio = shoulder_width / hip_width
            else:
                wh_ratio = 1.0

            # 宽高比 + 肩线角度 综合判断
            if wh_ratio > cfg.shoulder_hip_ratio_front and shoulder_angle < cfg.facing_front_thresh:
                # 肩宽明显大于髋宽，且肩线接近水平 -> 正面
                body_facing = "front"
                body_angle = shoulder_angle
                confidence = min(1.0, (wh_ratio - 1.0) * 1.5)

            elif wh_ratio < cfg.shoulder_hip_ratio_back:
                # 肩宽小于髋宽 -> 背面 (现实中不太可能，但作为判断条件)
                body_facing = "back"
                body_angle = 180.0 - shoulder_angle
                confidence = min(1.0, (1.0 - wh_ratio) * 2.0)

            else:
                # 肩宽与髋宽接近 -> 侧面，需要判断左右
                # 使用鼻子位置辅助判断
                if nose_vis and ls_vis and rs_vis:
                    # 鼻子相对于肩中点的水平位置
                    shoulder_center_x = (ls[0] + rs[0]) / 2
                    nose_offset = nose[0] - shoulder_center_x

                    # 肩线方向的x分量
                    if shoulder_vec[0] > 0:  # 正常肩线方向
                        if nose_offset > 0:
                            body_facing = "right"  # 右侧朝向镜头
                        else:
                            body_facing = "left"   # 左侧朝向镜头
                    else:
                        if nose_offset > 0:
                            body_facing = "left"
                        else:
                            body_facing = "right"

                    body_angle = 90.0
                    confidence = min(1.0, abs(nose_offset) / (shoulder_width * 0.5 + 1e-6))
                else:
                    # 无鼻子关键点，使用肩线倾斜方向判断
                    if shoulder_vec[0] > 0:
                        body_facing = "left" if shoulder_vec[1] > 0 else "right"
                    else:
                        body_facing = "right" if shoulder_vec[1] > 0 else "left"
                    body_angle = 90.0
                    confidence = 0.5

        elif ls_vis or rs_vis:
            # 只有部分关键点可见
            body_facing = "unknown"
            confidence = 0.3

        # 历史平滑
        self._ori_history.append((body_facing, body_angle, confidence))
        if len(self._ori_history) >= 2:
            body_facing, body_angle, confidence = self._smooth_orientation()

        # ========== 4. 枪口朝向估算 ==========
        muzzle_dir, muzzle_angle = self._estimate_muzzle_direction(keypoints)

        result = {
            "body_facing": body_facing,
            "body_angle": float(body_angle),
            "muzzle_dir": muzzle_dir,
            "muzzle_angle": float(muzzle_angle),
            "shoulder_width": float(shoulder_width),
            "hip_width": float(hip_width),
            "torso_tilt": float(torso_tilt),
            "confidence": float(confidence),
        }

        return result

    def _estimate_muzzle_direction(self, keypoints: np.ndarray) -> Tuple[str, float]:
        """
        估算枪口朝向 (基于手臂姿态)
        
        原理: 通过分析肩膀-手肘-手腕的夹角和方向，
        估算武器大致指向哪个方向
        
        Returns:
            (muzzle_direction, muzzle_angle)
        """
        # 关键点索引
        # 5: left_shoulder, 6: right_shoulder
        # 7: left_elbow, 8: right_elbow
        # 9: left_wrist, 10: right_wrist

        left_arm_valid = (
            keypoints[5][2] > self.cfg.kpt_visible_thresh and
            keypoints[7][2] > self.cfg.kpt_visible_thresh and
            keypoints[9][2] > self.cfg.kpt_visible_thresh
        )
        right_arm_valid = (
            keypoints[6][2] > self.cfg.kpt_visible_thresh and
            keypoints[8][2] > self.cfg.kpt_visible_thresh and
            keypoints[10][2] > self.cfg.kpt_visible_thresh
        )

        if not left_arm_valid and not right_arm_valid:
            return "unknown", 0.0

        # 计算手臂方向向量
        arm_angles = []

        if left_arm_valid:
            # 左手臂: shoulder(5) -> elbow(7) -> wrist(9)
            shoulder = keypoints[5][:2]
            elbow = keypoints[7][:2]
            wrist = keypoints[9][:2]

            # 大臂向量
            upper_arm = elbow - shoulder
            # 小臂向量
            forearm = wrist - elbow

            # 计算手臂整体方向 (加权平均)
            arm_dir = 0.6 * forearm + 0.4 * upper_arm
            arm_angle = math.degrees(math.atan2(arm_dir[1], arm_dir[0]))
            arm_angles.append(arm_angle)

        if right_arm_valid:
            shoulder = keypoints[6][:2]
            elbow = keypoints[8][:2]
            wrist = keypoints[10][:2]

            upper_arm = elbow - shoulder
            forearm = wrist - elbow

            arm_dir = 0.6 * forearm + 0.4 * upper_arm
            arm_angle = math.degrees(math.atan2(arm_dir[1], arm_dir[0]))
            arm_angles.append(arm_angle)

        # 平均手臂角度
        avg_angle = sum(arm_angles) / len(arm_angles)

        # 将角度转换为方向分类
        # 角度定义: 0=右, 90=下, 180=左, -90=上 (标准极坐标)
        # 但图像坐标系y轴向下，所以需要调整
        normalized_angle = avg_angle % 360
        if normalized_angle < 0:
            normalized_angle += 360

        # 转换为8方向分类
        if 315 <= normalized_angle or normalized_angle < 45:
            muzzle_dir = "right"
        elif 45 <= normalized_angle < 135:
            muzzle_dir = "down"  # 图像坐标系y向下
        elif 135 <= normalized_angle < 225:
            muzzle_dir = "left"
        else:
            muzzle_dir = "up"

        # 如果是双手前伸持枪姿势，判定为forward
        if left_arm_valid and right_arm_valid:
            # 检查双手是否都向前伸出
            ls = keypoints[5][:2]
            le = keypoints[7][:2]
            lw = keypoints[9][:2]
            rs = keypoints[6][:2]
            re = keypoints[8][:2]
            rw = keypoints[10][:2]

            # 计算手腕是否在手肘外侧 (远离身体)
            left_extended = np.linalg.norm(lw - ls) > np.linalg.norm(le - ls) * 1.5
            right_extended = np.linalg.norm(rw - rs) > np.linalg.norm(re - rs) * 1.5

            if left_extended and right_extended:
                muzzle_dir = "forward"

        return muzzle_dir, avg_angle

    def _smooth_orientation(self) -> Tuple[str, float, float]:
        """对历史朝向进行平滑"""
        if len(self._ori_history) == 0:
            return "unknown", 90.0, 0.0

        # 对角度进行平均
        angles = [h[1] for h in self._ori_history]
        confidences = [h[2] for h in self._ori_history]

        avg_angle = sum(angles) / len(angles)
        avg_conf = sum(confidences) / len(confidences)

        # 投票决定朝向类别
        facings = [h[0] for h in self._ori_history]
        facing_counts = {}
        for f in facings:
            facing_counts[f] = facing_counts.get(f, 0) + 1
        best_facing = max(facing_counts, key=facing_counts.get)

        return best_facing, avg_angle, avg_conf

    def _default_orientation(self) -> Dict:
        """返回默认朝向"""
        return {
            "body_facing": "unknown",
            "body_angle": 90.0,
            "muzzle_dir": "unknown",
            "muzzle_angle": 0.0,
            "shoulder_width": 0.0,
            "hip_width": 0.0,
            "torso_tilt": 0.0,
            "confidence": 0.0,
        }

    def reset(self):
        """重置历史记录"""
        self._ori_history.clear()
        self._muzzle_history.clear()


def compute_orientation_score(orientation: Dict, prioritize_back: bool = True) -> float:
    """
    根据身体朝向计算目标优先级分数
    
    Args:
        orientation: BodyOrientationEstimator.estimate()的输出
        prioritize_back: 是否优先击杀背身目标
        
    Returns:
        score: 0.0-1.0 的优先级分数 (越高越优先)
    """
    cfg = BODY_ORI_CFG
    facing = orientation.get("body_facing", "unknown")
    conf = orientation.get("confidence", 0.0)

    if facing == "front":
        # 正面朝向 - 威胁最高，优先击杀
        base_score = cfg.facing_front_weight
    elif facing in ("left", "right"):
        # 侧面
        base_score = cfg.facing_side_weight
    elif facing == "back":
        # 背面
        if prioritize_back:
            # 背身最容易击杀，提高优先级
            base_score = cfg.facing_front_weight * 1.1
        else:
            base_score = cfg.facing_back_weight
    else:
        # 未知朝向
        base_score = 0.5

    # 乘以置信度
    return base_score * conf
