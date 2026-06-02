"""
压枪补偿模块 (Recoil Compensator)
实现自动识别开火状态并施加反向鼠标偏移以抵消武器后坐力

核心算法:
1. 开火检测: 通过自动开火信号 + 目标命中状态判断
2. 固定模式压枪: 预定义每种武器的后坐力模式，按发数施加反向补偿
3. 自适应压枪: 根据实际命中位置反馈动态调整后坐力补偿量
4. 枪口回降模拟: 停止开火后，模拟枪口自然回落过程

学术参考:
- FPS游戏后坐力模式分析 (基于《Secrets of Gosu》论文中的职业玩家数据)
- 低秩近似检测合成后坐力控制 (Application of Low-Rank Approximation via SVD)
"""
import logging
import time
import random
from typing import Tuple, Dict, List, Optional, Deque
from collections import deque
import numpy as np

from src.config import RECOIL_CFG, AIMING_CFG

logger = logging.getLogger(__name__)


class RecoilPattern:
    """
    单武器后坐力模式
    存储预定义的后坐力偏移序列
    """

    def __init__(self, weapon_name: str, profile: Dict):
        self.weapon_name = weapon_name
        self.profile = profile
        self.vertical_per_shot = profile.get("vertical_per_shot", 2.5)
        self.horizontal_drift = profile.get("horizontal_drift", 0.3)
        self.max_compensation = profile.get("max_compensation", 25.0)
        self.recovery_rate = profile.get("recovery_rate", 0.3)
        self.bullets_per_pattern = profile.get("bullets_per_pattern", 30)

        # 预生成后坐力模式 (垂直方向累计偏移)
        self._pattern_y = []
        self._pattern_x = []
        self._generate_pattern()

    def _generate_pattern(self):
        """生成后坐力模式"""
        # 垂直偏移: 每发子弹增加一定的垂直上升
        # 前5-8发上升最快，之后趋于稳定
        cumulative_y = 0.0

        for i in range(self.bullets_per_pattern):
            # 垂直偏移量 (非线性: 前快后慢)
            if i < 8:
                shot_rise = self.vertical_per_shot * (1.0 - i * 0.08)
            else:
                shot_rise = self.vertical_per_shot * 0.4  # 后期稳定

            # 水平偏移 (随机左右漂移)
            if i < 10:
                shot_drift = random.uniform(-self.horizontal_drift, self.horizontal_drift)
            else:
                # 后期水平漂移加大 (控枪难度增加)
                shot_drift = random.uniform(-self.horizontal_drift * 2, self.horizontal_drift * 2)

            cumulative_y += shot_rise

            self._pattern_y.append(shot_rise)
            self._pattern_x.append(shot_drift)

    def get_compensation(self, shot_number: int) -> Tuple[float, float]:
        """
        获取第N发子弹的补偿偏移量
        
        Returns:
            (comp_x, comp_y): 补偿偏移 (反向，用于抵消后坐力)
        """
        if shot_number <= 0 or shot_number > len(self._pattern_y):
            return 0.0, 0.0

        idx = shot_number - 1
        comp_y = -self._pattern_y[idx]  # 反向补偿
        comp_x = -self._pattern_x[idx]

        # 限制最大补偿量
        comp_y = max(-self.max_compensation, min(self.max_compensation, comp_y))
        comp_x = max(-self.max_compensation, min(self.max_compensation, comp_x))

        return comp_x, comp_y

    def get_cumulative_offset(self, shot_number: int) -> Tuple[float, float]:
        """获取累计后坐力偏移"""
        if shot_number <= 0:
            return 0.0, 0.0

        total_y = sum(self._pattern_y[:shot_number])
        total_x = sum(self._pattern_x[:shot_number])
        return total_x, total_y


class RecoilCompensator:
    """
    压枪补偿器
    管理开火状态、应用后坐力补偿、枪口回降
    """

    def __init__(self):
        self.cfg = RECOIL_CFG

        # 武器模式库
        self.patterns: Dict[str, RecoilPattern] = {}
        self._load_weapon_profiles()

        # 当前状态
        self.current_weapon = self.cfg.current_weapon
        self.is_firing = False
        self.shot_counter = 0
        self.total_shots = 0
        self.consecutive_hits = 0

        # 当前累计补偿偏移 (用于枪口回降)
        self.current_comp_x = 0.0
        self.current_comp_y = 0.0

        # 枪口回降状态
        self.is_recovering = False
        self.recovery_target_x = 0.0
        self.recovery_target_y = 0.0

        # 自适应压枪数据
        self.adaptive_offsets_y: Deque[float] = deque(maxlen=self.cfg.adaptive_window_size)
        self.adaptive_offsets_x: Deque[float] = deque(maxlen=self.cfg.adaptive_window_size)
        self.adaptive_factor = 1.0  # 自适应系数 (1.0=标准)

        # 开火时间记录
        self.last_fire_time = 0.0
        self.fire_start_time = 0.0

        # 上一帧的瞄准偏移 (用于检测实际偏移)
        self.last_aim_dx = 0.0
        self.last_aim_dy = 0.0

    def _load_weapon_profiles(self):
        """加载武器配置文件"""
        for weapon_name, profile in self.cfg.weapon_profiles.items():
            self.patterns[weapon_name] = RecoilPattern(weapon_name, profile)
        logger.info(f"已加载 {len(self.patterns)} 种武器后坐力模式")

    def set_weapon(self, weapon_name: str):
        """切换当前武器"""
        if weapon_name in self.patterns:
            self.current_weapon = weapon_name
            logger.debug(f"切换武器: {weapon_name}")
        else:
            logger.warning(f"未知武器 '{weapon_name}'，使用默认配置")
            self.current_weapon = "default"

    def on_fire_start(self):
        """检测到开火开始"""
        if not self.cfg.enabled:
            return

        self.is_firing = True
        self.is_recovering = False
        self.fire_start_time = time.time()
        self.shot_counter = 0

        logger.debug("开火开始")

    def on_fire_stop(self):
        """检测到开火停止"""
        if not self.cfg.enabled:
            return

        self.is_firing = False
        self.is_recovering = True
        self.recovery_target_x = 0.0
        self.recovery_target_y = 0.0
        self.last_fire_time = time.time()
        self.consecutive_hits = 0

        # 记录本次连射数据用于自适应学习
        if self.cfg.mode in ("adaptive", "hybrid") and self.total_shots > 5:
            self._update_adaptive_factor()

        logger.debug(f"开火停止，本次连射 {self.total_shots} 发")
        self.total_shots = 0

    def on_shot_fired(self, hit_target: bool = False):
        """
        检测到单发子弹发射
        
        Args:
            hit_target: 是否命中目标
        """
        if not self.cfg.enabled or not self.is_firing:
            return

        self.shot_counter += 1
        self.total_shots += 1

        if hit_target:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0

        # 限制最大压枪发数
        if self.shot_counter > self.cfg.max_shots_compensated:
            self.shot_counter = self.cfg.max_shots_compensated

    def get_compensation_offset(self, current_dx: float, current_dy: float) -> Tuple[float, float]:
        """
        获取当前帧的压枪补偿偏移量
        
        Args:
            current_dx: 当前帧的水平瞄准偏移
            current_dy: 当前帧的垂直瞄准偏移
            
        Returns:
            (comp_offset_x, comp_offset_y): 需要额外施加的压枪补偿偏移
        """
        if not self.cfg.enabled:
            return 0.0, 0.0

        self.last_aim_dx = current_dx
        self.last_aim_dy = current_dy

        # ===== 枪口回降处理 =====
        if self.is_recovering and not self.is_firing:
            return self._process_recovery()

        # ===== 开火状态压枪 =====
        if self.is_firing and self.shot_counter > self.cfg.fire_detect_threshold:
            return self._compute_firing_compensation()

        return 0.0, 0.0

    def _compute_firing_compensation(self) -> Tuple[float, float]:
        """计算开火状态的压枪补偿"""
        pattern = self.patterns.get(self.current_weapon, self.patterns["default"])

        # 获取基础补偿
        base_comp_x, base_comp_y = pattern.get_compensation(self.shot_counter)

        # 应用自适应系数
        if self.cfg.mode in ("adaptive", "hybrid"):
            base_comp_y *= self.adaptive_factor
            base_comp_x *= self.adaptive_factor

        # 累积补偿
        self.current_comp_x += base_comp_x
        self.current_comp_y += base_comp_y

        # 记录用于自适应学习
        self.adaptive_offsets_x.append(base_comp_x)
        self.adaptive_offsets_y.append(base_comp_y)

        return base_comp_x, base_comp_y

    def _process_recovery(self) -> Tuple[float, float]:
        """
        处理枪口回降
        停止开火后，逐渐将补偿偏移恢复到0
        """
        pattern = self.patterns.get(self.current_weapon, self.patterns["default"])
        recovery = pattern.recovery_rate

        # 线性回降
        recovery_x = -self.current_comp_x * recovery
        recovery_y = -self.current_comp_y * recovery

        self.current_comp_x += recovery_x
        self.current_comp_y += recovery_y

        # 判断是否回降完成
        if abs(self.current_comp_x) < 0.5 and abs(self.current_comp_y) < 0.5:
            self.current_comp_x = 0.0
            self.current_comp_y = 0.0
            self.is_recovering = False
            logger.debug("枪口回降完成")

        return recovery_x, recovery_y

    def _update_adaptive_factor(self):
        """
        更新自适应系数
        根据实际命中情况调整压枪强度
        """
        if len(self.adaptive_offsets_y) < 5:
            return

        # 分析最近命中的散布情况
        # 如果命中率高但散布大 -> 增加补偿
        # 如果命中率低 -> 减少补偿或调整方向
        avg_offset_y = np.mean(list(self.adaptive_offsets_y))

        if self.consecutive_hits >= 3:
            # 连续命中，补偿合适或略弱
            self.adaptive_factor = min(1.2, self.adaptive_factor + self.cfg.adaptive_learning_rate)
        elif self.consecutive_hits == 0:
            # 连续未命中，补偿可能过强或方向不对
            self.adaptive_factor = max(0.7, self.adaptive_factor - self.cfg.adaptive_learning_rate * 2)

        self.adaptive_factor = np.clip(self.adaptive_factor, 0.5, 1.5)

        logger.debug(f"自适应系数更新: {self.adaptive_factor:.3f}")

    def get_weapon_list(self) -> List[str]:
        """获取支持的武器列表"""
        return list(self.patterns.keys())

    def get_status(self) -> Dict:
        """获取当前压枪状态信息"""
        return {
            "enabled": self.cfg.enabled,
            "is_firing": self.is_firing,
            "is_recovering": self.is_recovering,
            "weapon": self.current_weapon,
            "shot_counter": self.shot_counter,
            "adaptive_factor": self.adaptive_factor,
            "current_comp_x": self.current_comp_x,
            "current_comp_y": self.current_comp_y,
        }

    def reset(self):
        """重置压枪状态"""
        self.is_firing = False
        self.is_recovering = False
        self.shot_counter = 0
        self.total_shots = 0
        self.consecutive_hits = 0
        self.current_comp_x = 0.0
        self.current_comp_y = 0.0
        self.adaptive_factor = 1.0
        self.adaptive_offsets_x.clear()
        self.adaptive_offsets_y.clear()
