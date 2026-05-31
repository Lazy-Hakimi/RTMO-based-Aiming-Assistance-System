"""
RTMO 模型输出解码器 (增强版)
新增功能: 身体朝向信息可视化、准星校准UI叠加
"""
import logging
from typing import List, Tuple, Dict
import numpy as np
import cv2

from src.config import MODEL_CFG, AIMING_CFG, COCO_KEYPOINTS, BODY_ORI_CFG

logger = logging.getLogger(__name__)


class RTMODecoder:
    """RTMO 输出解码器 (原版逻辑完全保留)"""

    def __init__(self,
                 conf_thresh: float = 0.3,
                 nms_thresh: float = 0.65,
                 max_detections: int = 10,
                 num_keypoints: int = 17):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.num_keypoints = num_keypoints

        self.shoulder_pts = [5, 6]
        self.hip_pts = [11, 12]
        self.head_pts = [0, 1, 2, 3, 4]

    def decode(self,
               outputs: List[np.ndarray],
               scale: float,
               pad_offset: Tuple[int, int],
               orig_shape: Tuple[int, int]) -> List[Dict]:
        if len(outputs) == 2:
            dets, kpts = self._decode_standard(outputs, scale, pad_offset, orig_shape)
        elif len(outputs) == 3:
            dets, kpts = self._decode_yolox(outputs, scale, pad_offset, orig_shape)
        else:
            dets, kpts = self._decode_end2end(outputs[0], scale, pad_offset, orig_shape)

        persons = []
        for i in range(len(dets)):
            bbox = dets[i][:4]
            score = dets[i][4]
            keypoints = kpts[i] if i < len(kpts) else np.zeros((self.num_keypoints, 3))

            if score < self.conf_thresh:
                continue

            person_height = self._estimate_person_height(keypoints)
            if person_height > 0:
                height_ratio = person_height / orig_shape[0]
                if height_ratio < AIMING_CFG.min_person_height_ratio or \
                   height_ratio > AIMING_CFG.max_person_height_ratio:
                    continue

            persons.append({
                "bbox": bbox,
                "score": float(score),
                "keypoints": keypoints,
                "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                "height": person_height
            })

        persons.sort(key=lambda x: x["score"], reverse=True)
        return persons[:self.max_detections]

    def _decode_standard(self, outputs, scale, pad_offset, orig_shape):
        dets_raw = outputs[0]
        kpts_raw = outputs[1]
        if dets_raw.ndim == 3:
            dets_raw = dets_raw[0]
        if kpts_raw.ndim == 4:
            kpts_raw = kpts_raw[0]

        mask = dets_raw[:, 4] > self.conf_thresh
        dets = dets_raw[mask]
        kpts = kpts_raw[mask]

        if len(dets) > 0:
            indices = self._nms(dets[:, :4], dets[:, 4], self.nms_thresh)
            dets = dets[indices]
            kpts = kpts[indices]

        pad_x, pad_y = pad_offset
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / scale

        h, w = orig_shape
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h)
        kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w)
        kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h)

        return dets, kpts

    def _decode_yolox(self, outputs, scale, pad_offset, orig_shape):
        bboxes = outputs[0]
        scores = outputs[1]
        kpts = outputs[2]

        if bboxes.ndim == 3:
            bboxes = bboxes[0]
            scores = scores[0]
            kpts = kpts[0]

        if scores.shape[1] > 1:
            class_scores = scores[:, 1:]
            class_ids = np.argmax(class_scores, axis=1)
            conf = np.max(class_scores, axis=1)
        else:
            conf = scores[:, 0]
            class_ids = np.zeros(len(conf), dtype=np.int32)

        mask = conf > self.conf_thresh
        bboxes = bboxes[mask]
        conf = conf[mask]
        class_ids = class_ids[mask]
        kpts = kpts[mask]

        kpts = kpts.reshape(-1, self.num_keypoints, 3)

        dets = np.concatenate([
            bboxes,
            conf.reshape(-1, 1),
            class_ids.reshape(-1, 1)
        ], axis=1)

        if len(dets) > 0:
            indices = self._nms(dets[:, :4], dets[:, 4], self.nms_thresh)
            dets = dets[indices]
            kpts = kpts[indices]

        pad_x, pad_y = pad_offset
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / scale

        h, w = orig_shape
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h)
        kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w)
        kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h)

        return dets, kpts

    def _decode_end2end(self, output, scale, pad_offset, orig_shape):
        if output.ndim == 3:
            output = output[0]

        dets = output[:, :6]
        kpts = output[:, 6:].reshape(-1, self.num_keypoints, 3)

        mask = dets[:, 4] > self.conf_thresh
        dets = dets[mask]
        kpts = kpts[mask]

        pad_x, pad_y = pad_offset
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale
        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / scale

        h, w = orig_shape
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h)
        kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w)
        kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h)

        return dets, kpts

    def _nms(self, boxes: np.ndarray, scores: np.ndarray, thresh: float) -> List[int]:
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

    def _estimate_person_height(self, keypoints: np.ndarray) -> float:
        valid = keypoints[:, 2] > AIMING_CFG.kpt_visible_thresh
        y_coords = []
        if valid[0]:
            y_coords.append(keypoints[0, 1])
        for pt in self.head_pts:
            if valid[pt]:
                y_coords.append(keypoints[pt, 1])
        lower_y = []
        lower_pts = [11, 12, 13, 14, 15, 16]
        for pt in lower_pts:
            if valid[pt]:
                lower_y.append(keypoints[pt, 1])
        if len(y_coords) > 0 and len(lower_y) > 0:
            return max(lower_y) - min(y_coords)
        return 0.0

    def get_aim_point(self, person: Dict) -> Tuple[float, float, float]:
        keypoints = person["keypoints"]
        for kp_idx in AIMING_CFG.priority_keypoints:
            if keypoints[kp_idx, 2] > AIMING_CFG.kpt_visible_thresh:
                return (keypoints[kp_idx, 0], keypoints[kp_idx, 1], keypoints[kp_idx, 2])
        valid_kpts = []
        for kp_idx in AIMING_CFG.fallback_keypoints:
            if keypoints[kp_idx, 2] > AIMING_CFG.kpt_visible_thresh:
                valid_kpts.append(keypoints[kp_idx])
        if valid_kpts:
            valid_kpts = np.array(valid_kpts)
            cx = np.mean(valid_kpts[:, 0])
            cy = np.mean(valid_kpts[:, 1])
            conf = np.mean(valid_kpts[:, 2])
            return (cx, cy, conf)
        bbox = person["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[1] + (bbox[3] - bbox[1]) * 0.25
        return (cx, cy, person["score"] * 0.5)


# ============ 可视化函数 (增强版) ============

# 朝向对应的颜色
FACING_COLORS = {
    "front": (0, 255, 0),     # 正面 - 绿色
    "left": (255, 165, 0),    # 左侧 - 橙色
    "right": (255, 165, 0),   # 右侧 - 橙色
    "back": (0, 165, 255),    # 背面 - 青色
    "unknown": (128, 128, 128), # 未知 - 灰色
}

# 朝向对应的中文标签
FACING_LABELS = {
    "front": "FRONT",
    "left": "LEFT",
    "right": "RIGHT",
    "back": "BACK",
    "unknown": "?",
}


def draw_debug_info(image: np.ndarray,
                    persons: List[Dict],
                    target_person: Dict = None,
                    aim_point: Tuple[float, float] = None,
                    latency_ms: float = 0,
                    show_orientation: bool = True,
                    calibrator_info: Dict = None,
                    recoil_info: Dict = None) -> np.ndarray:
    """
    绘制增强版调试信息
    
    新增:
    - 身体朝向指示箭头
    - 优先级分数显示
    - 压枪状态指示
    - 准星校准UI (激活时)
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for person in persons:
        bbox = person["bbox"].astype(int)
        score = person["score"]
        kpts = person["keypoints"]

        # 判断是否为目标
        is_target = (person is target_person)

        # 获取朝向信息
        orientation = person.get("body_orientation", {})
        facing = orientation.get("body_facing", "unknown") if show_orientation else "unknown"

        # 颜色: 目标用红色，其他根据朝向
        if is_target:
            color = (0, 0, 255)  # 红色 - 目标
        else:
            color = FACING_COLORS.get(facing, (0, 255, 0))

        # 绘制边界框
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # 绘制置信度
        cv2.putText(vis, f"{score:.2f}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制朝向标签
        if show_orientation and facing != "unknown":
            ori_label = FACING_LABELS.get(facing, "?")
            ori_conf = orientation.get("confidence", 0)
            label_text = f"{ori_label}({ori_conf:.1f})"

            # 标签背景
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (bbox[0], bbox[3]), (bbox[0] + tw + 4, bbox[3] + th + 4),
                          color, -1)
            cv2.putText(vis, label_text, (bbox[0] + 2, bbox[3] + th + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # 绘制朝向箭头
        if show_orientation and is_target and facing != "unknown":
            _draw_orientation_arrow(vis, person, orientation)

        # 绘制优先级分数
        priority_score = person.get("priority_score", None)
        if priority_score is not None and not is_target:
            score_text = f"P:{priority_score:.2f}"
            cv2.putText(vis, score_text, (bbox[2] - 50, bbox[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # 绘制关键点
        for i, (x, y, v) in enumerate(kpts):
            if v > AIMING_CFG.kpt_visible_thresh:
                if i in [0, 1, 2]:
                    c = (0, 255, 255)  # 头部 - 青色
                elif i in [5, 6]:
                    c = (255, 255, 0)  # 肩部 - 黄色
                elif i in [11, 12]:
                    c = (255, 0, 255)  # 髋部 - 紫色
                else:
                    c = (255, 0, 0)    # 其他 - 蓝色
                cv2.circle(vis, (int(x), int(y)), 3, c, -1)

        # 绘制骨架
        for (a, b) in [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                       (5, 11), (6, 12), (11, 12),
                       (11, 13), (13, 15), (12, 14), (14, 16)]:
            if kpts[a, 2] > 0.3 and kpts[b, 2] > 0.3:
                pt1 = (int(kpts[a, 0]), int(kpts[a, 1]))
                pt2 = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(vis, pt1, pt2, (128, 128, 255), 1)

    # 绘制瞄准点
    if aim_point is not None:
        ax, ay = int(aim_point[0]), int(aim_point[1])
        cv2.drawMarker(vis, (ax, ay), (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
        cv2.circle(vis, (ax, ay), 30, (0, 0, 255), 1)

    # 绘制延迟信息
    info_lines = [
        f"Latency: {latency_ms:.1f}ms | Persons: {len(persons)}",
    ]

    # 压枪状态
    if recoil_info and recoil_info.get("enabled"):
        firing = recoil_info.get("is_firing", False)
        shots = recoil_info.get("shot_counter", 0)
        weapon = recoil_info.get("weapon", "default")
        info_lines.append(f"Recoil: {'FIRING' if firing else 'idle'} | {weapon} | shots:{shots}")

    # 校准状态
    if calibrator_info and calibrator_info.get("is_calibrating"):
        offset_x = calibrator_info.get("offset_x", 0)
        offset_y = calibrator_info.get("offset_y", 0)
        info_lines.append(f"CALIBRATING | Offset: ({offset_x:.1f}, {offset_y:.1f})")

    # 绘制信息文本
    for i, text in enumerate(info_lines):
        y_pos = 30 + i * 25
        # 文字阴影
        cv2.putText(vis, text, (12, y_pos + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
        cv2.putText(vis, text, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

    # 绘制瞄准区域
    rx1 = int(AIMING_CFG.aim_region_x[0] * w)
    rx2 = int(AIMING_CFG.aim_region_x[1] * w)
    ry1 = int(AIMING_CFG.aim_region_y[0] * h)
    ry2 = int(AIMING_CFG.aim_region_y[1] * h)
    overlay = vis.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    # 绘制图例
    _draw_legend(vis)

    return vis


def _draw_orientation_arrow(image: np.ndarray, person: Dict, orientation: Dict):
    """绘制身体朝向指示箭头"""
    facing = orientation.get("body_facing", "unknown")
    if facing == "unknown":
        return

    bbox = person["bbox"]
    cx = int((bbox[0] + bbox[2]) / 2)
    cy = int(bbox[1] - 20)

    # 箭头方向
    arrow_len = 25
    dx, dy = 0, 0

    if facing == "front":
        dx, dy = 0, -arrow_len  # 向上 (面向镜头)
    elif facing == "back":
        dx, dy = 0, arrow_len   # 向下
    elif facing == "left":
        dx, dy = -arrow_len, 0  # 向左
    elif facing == "right":
        dx, dy = arrow_len, 0   # 向右

    color = FACING_COLORS.get(facing, (128, 128, 128))
    cv2.arrowedLine(image, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.3)

    # 标签
    label = FACING_LABELS.get(facing, "?")
    cv2.putText(image, label, (cx - 15, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def _draw_legend(image: np.ndarray):
    """绘制朝向图例"""
    h, w = image.shape[:2]
    legend_x = w - 120
    legend_y = h - 100

    # 背景
    overlay = image.copy()
    cv2.rectangle(overlay, (legend_x - 10, legend_y - 20),
                  (legend_x + 110, legend_y + 80), (0, 0, 0), -1)
    image[:] = cv2.addWeighted(image, 1.0, overlay, 0.7, 0)

    cv2.putText(image, "Orientation:", (legend_x, legend_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    items = [
        ("FRONT", FACING_COLORS["front"]),
        ("SIDE", FACING_COLORS["left"]),
        ("BACK", FACING_COLORS["back"]),
    ]

    for i, (label, color) in enumerate(items):
        y = legend_y + 20 + i * 18
        cv2.circle(image, (legend_x + 5, y - 3), 5, color, -1)
        cv2.putText(image, label, (legend_x + 15, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
