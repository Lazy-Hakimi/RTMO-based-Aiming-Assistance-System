"""
RTMO 模型输出解码器
将 TensorRT 输出解析为边界框 + 关键点
包含 NMS、坐标恢复、关键点可见性过滤
"""
import logging
from typing import List, Tuple, Dict
import numpy as np
import cv2

from src.config import MODEL_CFG, AIMING_CFG, COCO_KEYPOINTS

logger = logging.getLogger(__name__)


class RTMODecoder:
    """
    RTMO 输出解码器

    RTMO 基于 YOLOX 架构，输出包含：
    1. 边界框回归 (xyxy)
    2. 目标置信度 (objectness + classification)
    3. 关键点坐标 (通过动态坐标分类解码)
    4. 关键点可见性

    本解码器适配 ONNX 导出后的标准输出格式：
    - dets: [N, 6]  (x1, y1, x2, y2, conf, class)
    - keypoints: [N, 17, 3] (x, y, visibility)
    """

    def __init__(self, 
                 conf_thresh: float = 0.3,
                 nms_thresh: float = 0.65,
                 max_detections: int = 10,
                 num_keypoints: int = 17):
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.max_detections = max_detections
        self.num_keypoints = num_keypoints

        # COCO 关键点配对 (用于计算人体中心/高度)
        self.shoulder_pts = [5, 6]  # left_shoulder, right_shoulder
        self.hip_pts = [11, 12]     # left_hip, right_hip
        self.head_pts = [0, 1, 2, 3, 4]  # nose, eyes, ears

    def decode(self, 
               outputs: List[np.ndarray],
               scale: float,
               pad_offset: Tuple[int, int],
               orig_shape: Tuple[int, int]) -> List[Dict]:
        """
        解码模型输出
        Args:
            outputs: TensorRT 输出列表 (根据导出方式不同，格式可能不同)
            scale: 预处理缩放比例
            pad_offset: (pad_x, pad_y)
            orig_shape: 原始图像 (H, W)
        Returns:
            persons: 每个人包含 bbox, keypoints, score
        """
        # 根据输出数量判断导出格式
        if len(outputs) == 2:
            # 格式1: [dets, keypoints]
            dets, kpts = self._decode_standard(outputs, scale, pad_offset, orig_shape)
        elif len(outputs) == 3:
            # 格式2: [bboxes, scores, keypoints] (原始 YOLOX 格式)
            dets, kpts = self._decode_yolox(outputs, scale, pad_offset, orig_shape)
        else:
            # 格式3: 单输出端到端 (需要自定义解析)
            dets, kpts = self._decode_end2end(outputs[0], scale, pad_offset, orig_shape)

        # 组装结果
        persons = []
        for i in range(len(dets)):
            bbox = dets[i][:4]
            score = dets[i][4]
            keypoints = kpts[i] if i < len(kpts) else np.zeros((self.num_keypoints, 3))

            # 过滤低置信度人体
            if score < self.conf_thresh:
                continue

            # 计算人体高度比例 (用于过滤误检)
            person_height = self._estimate_person_height(keypoints)
            if person_height > 0:
                height_ratio = person_height / orig_shape[0]
                if height_ratio < AIMING_CFG.min_person_height_ratio or \
                   height_ratio > AIMING_CFG.max_person_height_ratio:
                    continue

            persons.append({
                "bbox": bbox,           # [x1, y1, x2, y2] 原图坐标
                "score": float(score),
                "keypoints": keypoints,  # [17, 3] (x, y, visibility)
                "center": ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
                "height": person_height
            })

        # 按置信度排序
        persons.sort(key=lambda x: x["score"], reverse=True)
        return persons[:self.max_detections]

    def _decode_standard(self, outputs, scale, pad_offset, orig_shape):
        """标准导出格式: [dets, keypoints]"""
        dets_raw = outputs[0]  # [1, N, 6] or [N, 6]
        kpts_raw = outputs[1]  # [1, N, 17, 3] or [N, 17, 3]

        # 去除 batch 维度
        if dets_raw.ndim == 3:
            dets_raw = dets_raw[0]
        if kpts_raw.ndim == 4:
            kpts_raw = kpts_raw[0]

        # 过滤低置信度
        mask = dets_raw[:, 4] > self.conf_thresh
        dets = dets_raw[mask]
        kpts = kpts_raw[mask]

        # NMS
        if len(dets) > 0:
            indices = self._nms(dets[:, :4], dets[:, 4], self.nms_thresh)
            dets = dets[indices]
            kpts = kpts[indices]

        # 坐标映射回原图 (去除 letterbox padding 和缩放)
        pad_x, pad_y = pad_offset
        dets[:, [0, 2]] = (dets[:, [0, 2]] - pad_x) / scale
        dets[:, [1, 3]] = (dets[:, [1, 3]] - pad_y) / scale

        kpts[:, :, 0] = (kpts[:, :, 0] - pad_x) / scale
        kpts[:, :, 1] = (kpts[:, :, 1] - pad_y) / scale

        # 裁剪到图像边界
        h, w = orig_shape
        dets[:, [0, 2]] = np.clip(dets[:, [0, 2]], 0, w)
        dets[:, [1, 3]] = np.clip(dets[:, [1, 3]], 0, h)
        kpts[:, :, 0] = np.clip(kpts[:, :, 0], 0, w)
        kpts[:, :, 1] = np.clip(kpts[:, :, 1], 0, h)

        return dets, kpts

    def _decode_yolox(self, outputs, scale, pad_offset, orig_shape):
        """YOLOX 原始格式: [bboxes, scores, keypoints]"""
        bboxes = outputs[0]  # [1, N, 4]
        scores = outputs[1]  # [1, N, num_classes+1]
        kpts = outputs[2]    # [1, N, num_keypoints*3]

        if bboxes.ndim == 3:
            bboxes = bboxes[0]
            scores = scores[0]
            kpts = kpts[0]

        # 取最高类别置信度
        if scores.shape[1] > 1:
            class_scores = scores[:, 1:]  # 排除背景类
            class_ids = np.argmax(class_scores, axis=1)
            conf = np.max(class_scores, axis=1)
        else:
            conf = scores[:, 0]
            class_ids = np.zeros(len(conf), dtype=np.int32)

        # 过滤
        mask = conf > self.conf_thresh
        bboxes = bboxes[mask]
        conf = conf[mask]
        class_ids = class_ids[mask]
        kpts = kpts[mask]

        # 重塑关键点
        kpts = kpts.reshape(-1, self.num_keypoints, 3)

        # 组装 dets [x1, y1, x2, y2, conf, class]
        dets = np.concatenate([
            bboxes,
            conf.reshape(-1, 1),
            class_ids.reshape(-1, 1)
        ], axis=1)

        # NMS
        if len(dets) > 0:
            indices = self._nms(dets[:, :4], dets[:, 4], self.nms_thresh)
            dets = dets[indices]
            kpts = kpts[indices]

        # 坐标映射
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
        """端到端导出格式 (含 NMS 后处理)"""
        # 假设输出为 [N, 6 + 17*3] = [N, 57]
        if output.ndim == 3:
            output = output[0]

        dets = output[:, :6]
        kpts = output[:, 6:].reshape(-1, self.num_keypoints, 3)

        # 过滤
        mask = dets[:, 4] > self.conf_thresh
        dets = dets[mask]
        kpts = kpts[mask]

        # 坐标映射
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
        """CPU NMS 实现"""
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
        """通过关键点估算人体高度 (像素)"""
        # 优先使用 shoulder -> hip -> knee -> ankle
        valid = keypoints[:, 2] > AIMING_CFG.kpt_visible_thresh

        y_coords = []
        if valid[0]:  # nose
            y_coords.append(keypoints[0, 1])

        # 找最高可见点 (通常是头部)
        for pt in self.head_pts:
            if valid[pt]:
                y_coords.append(keypoints[pt, 1])

        # 找最低可见点
        lower_pts = [11, 12, 13, 14, 15, 16]  # hip, knee, ankle
        lower_y = []
        for pt in lower_pts:
            if valid[pt]:
                lower_y.append(keypoints[pt, 1])

        if len(y_coords) > 0 and len(lower_y) > 0:
            return max(lower_y) - min(y_coords)

        # fallback: 使用 bbox 高度
        return 0.0

    def get_aim_point(self, person: Dict) -> Tuple[float, float, float]:
        """
        获取瞄准点坐标
        Returns:
            (x, y, confidence)  如果找不到有效点返回 (-1, -1, 0)
        """
        keypoints = person["keypoints"]

        # 1. 优先使用头部关键点
        for kp_idx in AIMING_CFG.priority_keypoints:
            if keypoints[kp_idx, 2] > AIMING_CFG.kpt_visible_thresh:
                return (keypoints[kp_idx, 0], keypoints[kp_idx, 1], keypoints[kp_idx, 2])

        # 2. Fallback 到躯干关键点
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

        # 3. 最终 fallback: 使用 bbox 中心上部 (模拟头部位置)
        bbox = person["bbox"]
        cx = (bbox[0] + bbox[2]) / 2
        cy = bbox[1] + (bbox[3] - bbox[1]) * 0.25  # 上部 25% 处
        return (cx, cy, person["score"] * 0.5)


def draw_debug_info(image: np.ndarray, 
                    persons: List[Dict],
                    target_person: Dict = None,
                    aim_point: Tuple[float, float] = None,
                    latency_ms: float = 0) -> np.ndarray:
    """
    绘制调试信息
    """
    vis = image.copy()
    h, w = vis.shape[:2]

    for person in persons:
        bbox = person["bbox"].astype(int)
        score = person["score"]
        kpts = person["keypoints"]

        # 绘制边界框
        color = (0, 255, 0) if person != target_person else (0, 0, 255)
        cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        cv2.putText(vis, f"{score:.2f}", (bbox[0], bbox[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 绘制关键点
        for i, (x, y, v) in enumerate(kpts):
            if v > AIMING_CFG.kpt_visible_thresh:
                c = (0, 255, 255) if i in [0, 1, 2] else (255, 0, 0)
                cv2.circle(vis, (int(x), int(y)), 3, c, -1)

        # 绘制骨架
        for (a, b) in [(5,6), (5,7), (7,9), (6,8), (8,10), (5,11), (6,12), (11,12)]:
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
    info_text = f"Latency: {latency_ms:.1f}ms | Persons: {len(persons)}"
    cv2.putText(vis, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 绘制瞄准区域
    rx1 = int(AIMING_CFG.aim_region_x[0] * w)
    rx2 = int(AIMING_CFG.aim_region_x[1] * w)
    ry1 = int(AIMING_CFG.aim_region_y[0] * h)
    ry2 = int(AIMING_CFG.aim_region_y[1] * h)
    overlay = vis.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (255, 255, 0), -1)
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

    return vis
