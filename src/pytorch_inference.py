"""
PyTorch 原生推理引擎封装 (RTMO 模型)
支持直接使用官方 .pth 权重文件进行 FP16 推理
适配 Jetson AGX Xavier (ARM64 + CUDA)

功能:
1. 使用 MMPose init_model 加载 RTMO 官方 PyTorch 权重
2. 支持 FP16 半精度推理 (torch.cuda.amp.autocast)
3. 输出格式与 TensorRT 引擎完全兼容 (decoder 无需修改)
4. 与 TrtInferenceEngine 保持相同接口，无缝替换

依赖:
    - torch>=1.12.0
    - mmcv>=2.0.0
    - mmpose>=1.0.0
    - mmengine>=0.7.0
"""
import os
import time
import logging
from typing import Tuple, List, Optional

import numpy as np
import cv2
import torch
import torch.nn as nn

from src.config import MODEL_CFG, SYS_CFG

logger = logging.getLogger(__name__)

# 尝试导入 MMPose 相关库
try:
    from mmpose.apis import init_model
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint
    HAS_MMPOSE = True
except ImportError as e:
    HAS_MMPOSE = False
    logger.warning(f"MMPose 未安装，PyTorch 推理引擎不可用: {e}")
    logger.warning("安装命令: pip install mmpose mmcv mmengine")


class PytorchInferenceEngine:
    """
    PyTorch 原生推理引擎 (RTMO 模型)
    
    直接使用 MMPose 的 init_model 加载官方 .pth 权重，
    支持 FP16 半精度推理，输出格式与 TensorRT 引擎完全兼容。
    
    使用方式与 TrtInferenceEngine 完全一致:
        engine = PytorchInferenceEngine(config_path, checkpoint_path)
        outputs = engine.infer(preprocessed_image)
    """

    def __init__(self,
                 config_path: str,
                 checkpoint_path: str,
                 device: str = "cuda:0",
                 fp16: bool = True,
                 input_size: Tuple[int, int] = (640, 640)):
        """
        初始化 PyTorch 推理引擎
        
        Args:
            config_path: MMPose 配置文件路径 (.py)
            checkpoint_path: PyTorch 权重文件路径 (.pth)
            device: 推理设备 ('cuda:0' 或 'cpu')
            fp16: 是否启用 FP16 半精度推理
            input_size: 模型输入尺寸 (width, height)
        """
        if not HAS_MMPOSE:
            raise RuntimeError(
                "MMPose 未安装，无法使用 PyTorch 推理引擎。\n"
                "请安装: pip install mmpose mmcv mmengine"
            )

        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.fp16 = fp16
        self.input_size = input_size  # (W, H)
        self._model_loaded = False

        # CUDA stream (与 TensorRT 一致，支持异步推理)
        if "cuda" in device and torch.cuda.is_available():
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None
            self.device = "cpu"
            if fp16:
                logger.warning("CPU 设备不支持 FP16，已自动禁用")
                self.fp16 = False

        # 加载模型
        self._load_model()

        logger.info(f"PyTorch 引擎初始化完成:")
        logger.info(f"  配置文件: {config_path}")
        logger.info(f"  权重文件: {checkpoint_path}")
        logger.info(f"  设备: {self.device}")
        logger.info(f"  FP16: {self.fp16}")
        logger.info(f"  输入尺寸: {input_size}")

    def _load_model(self):
        """加载 MMPose RTMO 模型和权重"""
        t0 = time.time()
        logger.info("正在加载 PyTorch 模型...")

        # 检查文件存在性
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"MMPose 配置文件不存在: {self.config_path}\n"
                f"请从 MMPose 官方仓库下载对应的配置文件:\n"
                f"  configs/body_2d_keypoint/rtmo/body7/rtmo-l_16xb16-600e_coco-640x640.py"
            )
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"权重文件不存在: {self.checkpoint_path}\n"
                f"请从 MMPose Model Zoo 下载 RTMO-l 权重:\n"
                f"  https://download.openmmlab.com/mmpose/v1/projects/rtmo/"
                f"rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth"
            )

        try:
            # 使用 MMPose 的 init_model 加载模型
            self.model = init_model(
                self.config_path,
                self.checkpoint_path,
                device=self.device
            )
        except Exception as e:
            logger.error(f"init_model 加载失败，尝试备用方案: {e}")
            self._load_model_manual()

        # 设置为评估模式
        self.model.eval()

        # 启用 FP16 (半精度推理)
        if self.fp16:
            self.model.half()
            logger.info("模型已转换为 FP16 半精度模式")

        # 编译优化 (PyTorch 2.0+ 支持 torch.compile)
        if hasattr(torch, 'compile') and SYS_CFG.use_cuda_graph:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                logger.info("已启用 torch.compile 优化")
            except Exception as e:
                logger.warning(f"torch.compile 失败: {e}")

        self._model_loaded = True
        load_time = time.time() - t0
        logger.info(f"模型加载完成，耗时 {load_time:.2f} 秒")

    def _load_model_manual(self):
        """手动加载模型 (备用方案，当 init_model 失败时使用)"""
        from mmengine.config import Config
        from mmengine.registry import MODELS

        cfg = Config.fromfile(self.config_path)
        model_cfg = cfg.model

        # 构建模型
        self.model = MODELS.build(model_cfg)

        # 加载权重
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        logger.info("模型手动加载成功")

    def _prepare_input(self, input_image: np.ndarray) -> torch.Tensor:
        """
        将预处理后的 numpy 数组转换为 PyTorch Tensor
        
        Args:
            input_image: 预处理后的图像 [1, 3, H, W], float32
            
        Returns:
            torch.Tensor: [1, 3, H, W], 已移至目标设备
        """
        # numpy -> torch tensor
        if isinstance(input_image, np.ndarray):
            tensor = torch.from_numpy(input_image)
        else:
            tensor = input_image

        # 确保维度正确 [B, C, H, W]
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        # FP16 转换
        if self.fp16:
            tensor = tensor.half()

        return tensor

    def _postprocess_output(self, predictions) -> List[np.ndarray]:
        """
        将 MMPose 预测结果转换为与 TensorRT 兼容的输出格式
        
        RTMO 是 bottom-up 模型，输出包含检测框和关键点。
        需要转换为 decoder 支持的格式:
        - 标准格式: [dets[...,:6], kpts[...]] (len=2)
        - dets: [batch, num_dets, 6] -> [x1, y1, x2, y2, score, class]
        - kpts: [batch, num_dets, 17, 3] -> [x, y, visibility]
        
        Args:
            predictions: MMPose 预测结果 (PoseDataSample 列表)
            
        Returns:
            List[np.ndarray]: [dets_array, kpts_array]
        """
        if isinstance(predictions, list) and len(predictions) > 0:
            pred = predictions[0]
        else:
            pred = predictions

        # 从 PoseDataSample 提取预测结果
        try:
            pred_instances = pred.pred_instances
        except AttributeError:
            # 如果已经是 tensor 格式
            return self._postprocess_tensor_output(pred)

        if len(pred_instances) == 0:
            # 没有检测到人体，返回空数组
            empty_dets = np.zeros((1, 0, 6), dtype=np.float32)
            empty_kpts = np.zeros((1, 0, 17, 3), dtype=np.float32)
            return [empty_dets, empty_kpts]

        # 提取边界框 [num_dets, 4]
        bboxes = pred_instances.bboxes.cpu().numpy()  # [N, 4] (x1, y1, x2, y2)

        # 提取边界框置信度 [num_dets]
        if hasattr(pred_instances, 'bbox_scores'):
            bbox_scores = pred_instances.bbox_scores.cpu().numpy()
        else:
            # 使用关键点的平均分数作为 bbox score
            bbox_scores = np.ones(len(bboxes)) * 0.5

        # 提取关键点 [num_dets, 17, 2] 和 置信度 [num_dets, 17]
        keypoints = pred_instances.keypoints  # [N, 17, 2]
        if hasattr(pred_instances, 'keypoint_scores'):
            kpt_scores = pred_instances.keypoint_scores  # [N, 17]
        else:
            kpt_scores = np.ones((keypoints.shape[0], keypoints.shape[1]))

        # 转换为 numpy
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(kpt_scores, torch.Tensor):
            kpt_scores = kpt_scores.cpu().numpy()

        num_dets = len(bboxes)

        # 构建 dets: [num_dets, 6] -> [x1, y1, x2, y2, score, class_id]
        dets = np.zeros((num_dets, 6), dtype=np.float32)
        dets[:, :4] = bboxes
        dets[:, 4] = bbox_scores
        dets[:, 5] = 0  # class_id = 0 (person)

        # 构建 kpts: [num_dets, 17, 3] -> [x, y, visibility]
        kpts = np.zeros((num_dets, 17, 3), dtype=np.float32)
        kpts[:, :, :2] = keypoints
        kpts[:, :, 2] = kpt_scores

        # 添加 batch 维度 -> [1, num_dets, ...]
        dets = dets[np.newaxis, ...]  # [1, N, 6]
        kpts = kpts[np.newaxis, ...]  # [1, N, 17, 3]

        return [dets, kpts]

    def _postprocess_tensor_output(self, tensor_output) -> List[np.ndarray]:
        """
        处理纯 tensor 输出 (当模型直接返回 tensor 时使用)
        
        假设输出格式与 MMDeploy ONNX 导出一致:
        - end2end: [batch, num_dets, 6 + 17*3] -> [x1,y1,x2,y2,score,class,kpts...]
        """
        if isinstance(tensor_output, torch.Tensor):
            output = tensor_output.cpu().numpy()
        else:
            output = tensor_output

        if output.ndim == 2:
            output = output[np.newaxis, ...]  # 添加 batch 维度

        batch_size = output.shape[0]
        num_dets = output.shape[1]

        if num_dets == 0:
            empty_dets = np.zeros((batch_size, 0, 6), dtype=np.float32)
            empty_kpts = np.zeros((batch_size, 0, 17, 3), dtype=np.float32)
            return [empty_dets, empty_kpts]

        # 分离 dets 和 kpts
        dets = output[:, :, :6].astype(np.float32)  # [B, N, 6]
        kpts_flat = output[:, :, 6:]  # [B, N, 17*3]
        kpts = kpts_flat.reshape(batch_size, num_dets, 17, 3).astype(np.float32)

        return [dets, kpts]

    def infer(self, input_image: np.ndarray) -> List[np.ndarray]:
        """
        执行推理 (同步)
        
        Args:
            input_image: 预处理后的图像 [1, 3, H, W] 或 [3, H, W], float32
            
        Returns:
            outputs: 模型输出列表 [dets, kpts]，格式与 TensorRT 引擎一致
        """
        if not self._model_loaded:
            raise RuntimeError("模型尚未加载")

        # 准备输入
        input_tensor = self._prepare_input(input_image)

        # 推理
        with torch.no_grad():
            if self.fp16 and torch.cuda.is_available():
                with torch.cuda.amp.autocast():
                    predictions = self.model.test_step(
                        dict(inputs=input_tensor, data_samples=None)
                    )
            else:
                predictions = self.model.test_step(
                    dict(inputs=input_tensor, data_samples=None)
                )

        # 后处理: 转换为兼容格式
        outputs = self._postprocess_output(predictions)

        return outputs

    def infer_async(self, input_image: np.ndarray) -> List[np.ndarray]:
        """
        异步推理 (使用 CUDA Stream)
        
        注意: 当前 PyTorch 实现为同步返回，
        stream 可用于外部流水线并行。如需真正异步，
        请使用 torch.jit 或 torch.compile。
        
        Args:
            input_image: 预处理后的图像
            
        Returns:
            outputs: 模型输出列表
        """
        # PyTorch 异步推理需要配合 stream 使用
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                return self.infer(input_image)
        return self.infer(input_image)

    def synchronize(self):
        """同步 CUDA 流"""
        if self.stream is not None:
            self.stream.synchronize()
        elif torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_output_shapes(self) -> List[Tuple[str, Tuple, type]]:
        """
        获取输出形状信息 (与 TensorRT 引擎兼容)
        
        Returns:
            List[Tuple[str, Tuple, type]]: 输出张量信息列表
        """
        return [
            ("dets", (1, -1, 6), np.float32),      # [batch, num_dets, 6]
            ("keypoints", (1, -1, 17, 3), np.float32),  # [batch, num_dets, 17, 3]
        ]

    def release(self):
        """释放模型资源"""
        if hasattr(self, 'model'):
            del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("PyTorch 引擎资源已释放")

    def get_model_info(self) -> dict:
        """
        获取模型信息
        
        Returns:
            dict: 包含模型配置和状态信息的字典
        """
        info = {
            "engine_type": "PyTorch",
            "config_path": self.config_path,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "fp16": self.fp16,
            "input_size": self.input_size,
            "model_loaded": self._model_loaded,
        }

        # 尝试获取模型结构信息
        if self._model_loaded and hasattr(self.model, 'backbone'):
            try:
                import torchsummary
                info["has_torchsummary"] = True
            except ImportError:
                info["has_torchsummary"] = False

        return info


def create_pytorch_engine(config_path: Optional[str] = None,
                          checkpoint_path: Optional[str] = None,
                          device: str = "cuda:0",
                          fp16: bool = True) -> PytorchInferenceEngine:
    """
    工厂函数：创建 PyTorch 推理引擎
    
    Args:
        config_path: MMPose 配置文件路径，默认从 MODEL_CFG 读取
        checkpoint_path: PyTorch 权重路径，默认从 MODEL_CFG 读取
        device: 推理设备
        fp16: 是否启用 FP16
        
    Returns:
        PytorchInferenceEngine 实例
    """
    if config_path is None:
        config_path = MODEL_CFG.get("pytorch_config_path",
                                    "configs/rtmo-l_16xb16-600e_coco-640x640.py")
    if checkpoint_path is None:
        checkpoint_path = MODEL_CFG.get("pytorch_checkpoint_path",
                                         "models/rtmo-l_16xb16-600e_coco-640x640-6b10eda5_20231211.pth")

    return PytorchInferenceEngine(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
        fp16=fp16,
        input_size=(MODEL_CFG.input_width, MODEL_CFG.input_height)
    )
