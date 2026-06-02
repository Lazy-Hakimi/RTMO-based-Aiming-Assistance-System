#!/usr/bin/env python3
"""
鼠标灵敏度校准工具

使用方法：
1. 运行脚本：python3 scripts/calibrate_mouse.py
2. 在屏幕上移动鼠标，脚本会记录鼠标实际移动距离
3. 与画面中的目标移动距离对比，自动计算灵敏度系数

原理：
- 在 1920x1080 画面中，目标移动 100 像素
- 需要发送多少鼠标单位才能让准星也移动 100 像素
- 这个比例就是 sensitivity 系数
"""
import os
import sys
import time
import logging
import argparse
from collections import deque

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.mouse_hid import create_mouse_controller
from src.config import MOUSE_CFG, CAPTURE_CFG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MouseCalibrator:
    """鼠标灵敏度校准器"""

    def __init__(self, test_distance: int = 200, num_samples: int = 5):
        self.test_distance = test_distance  # 测试移动距离 (像素)
        self.num_samples = num_samples
        self.mouse = create_mouse_controller(dummy=False)

        # 记录数据
        self.sent_moves = []   # 发送的鼠标单位
        self.actual_moves = []  # 实际画面移动 (需要手动测量)

    def calibrate(self):
        """执行校准"""
        logger.info("=" * 50)
        logger.info("鼠标灵敏度校准")
        logger.info("=" * 50)
        logger.info(f"测试距离: {self.test_distance} 像素")
        logger.info(f"样本数: {self.num_samples}")
        logger.info("")
        logger.info("步骤:")
        logger.info("1. 进入游戏，将准星对准屏幕左侧某个固定参考点")
        logger.info("2. 按 Enter 开始自动向右移动")
        logger.info("3. 观察准星实际移动了多少像素")
        logger.info("4. 输入实际移动距离")
        logger.info("")

        input("按 Enter 开始校准...")

        for i in range(self.num_samples):
            logger.info(f"\n样本 {i+1}/{self.num_samples}")
            logger.info(f"发送鼠标移动: +{self.test_distance} 单位")

            # 发送移动
            self.mouse.move(self.test_distance, 0)
            time.sleep(0.5)

            # 获取用户输入的实际移动
            while True:
                try:
                    actual = input("准星实际移动了多少像素? ")
                    actual = float(actual)
                    if actual > 0:
                        break
                    logger.warning("请输入正数")
                except ValueError:
                    logger.warning("请输入数字")

            self.sent_moves.append(self.test_distance)
            self.actual_moves.append(actual)

            # 复位
            logger.info("复位中...")
            self.mouse.move(-self.test_distance, 0)
            time.sleep(0.5)

        # 计算灵敏度
        ratios = [s / a for s, a in zip(self.sent_moves, self.actual_moves)]
        avg_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)

        logger.info("\n" + "=" * 50)
        logger.info("校准结果:")
        logger.info(f"  平均灵敏度系数: {avg_ratio:.4f}")
        logger.info(f"  标准差: {std_ratio:.4f}")
        logger.info(f"  建议配置: sensitivity_x = {avg_ratio:.4f}")
        logger.info(f"  建议配置: sensitivity_y = {avg_ratio:.4f}")
        logger.info("=" * 50)

        # 保存配置
        self._save_config(avg_ratio)

        self.mouse.close()

    def _save_config(self, sensitivity: float):
        """保存校准结果到配置文件"""
        config_path = "configs/calibration.py"
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        with open(config_path, "w") as f:
            f.write(f"# 自动生成的校准配置\n")
            f.write(f"# 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"CALIBRATED_SENSITIVITY_X = {sensitivity:.4f}\n")
            f.write(f"CALIBRATED_SENSITIVITY_Y = {sensitivity:.4f}\n")

        logger.info(f"配置已保存: {config_path}")
        logger.info("请在 config.py 中更新 MOUSE_CFG.sensitivity_x/y")


class AutoCalibrator:
    """
    自动校准器 (使用 OpenCV 捕获画面检测准星移动)
    需要配合屏幕录制使用
    """

    def __init__(self, capture_device: str = "/dev/video0"):
        self.capture = cv2.VideoCapture(capture_device)
        self.mouse = create_mouse_controller(dummy=False)

    def auto_calibrate(self, test_distance: int = 100):
        """自动校准 (需要准星检测器)"""
        logger.info("自动校准模式")
        logger.info("请确保准星在画面中心可见")
        input("按 Enter 开始...")

        # 读取初始帧
        ret, frame0 = self.capture.read()
        if not ret:
            logger.error("无法读取画面")
            return

        h, w = frame0.shape[:2]
        cx, cy = w // 2, h // 2

        # 发送移动
        self.mouse.move(test_distance, 0)
        time.sleep(0.3)

        # 读取移动后帧
        ret, frame1 = self.capture.read()
        if not ret:
            logger.error("无法读取画面")
            return

        # 使用光流或模板匹配检测准星移动
        # 这里简化：使用帧差分
        diff = cv2.absdiff(frame0, frame1)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)

        # 找到变化区域
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 取最大轮廓的中心
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] > 0:
                new_cx = int(M["m10"] / M["m00"])
                actual_move = abs(new_cx - cx)
                sensitivity = test_distance / actual_move if actual_move > 0 else 1.0

                logger.info(f"检测到的移动: {actual_move} 像素")
                logger.info(f"灵敏度系数: {sensitivity:.4f}")

        self.capture.release()
        self.mouse.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                       help="使用自动校准 (需要画面检测)")
    parser.add_argument("--distance", type=int, default=200,
                       help="测试移动距离")
    parser.add_argument("--samples", type=int, default=5,
                       help="样本数量")
    parser.add_argument("--device", type=str, default="/dev/video0",
                       help="视频设备")

    args = parser.parse_args()

    if args.auto:
        calibrator = AutoCalibrator(args.device)
        calibrator.auto_calibrate(args.distance)
    else:
        calibrator = MouseCalibrator(args.distance, args.samples)
        calibrator.calibrate()


if __name__ == "__main__":
    main()
