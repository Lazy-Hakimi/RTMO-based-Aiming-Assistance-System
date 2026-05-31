"""
ONNX Runtime GPU 推理引擎封装
替代 TensorRT，直接运行 ONNX 模型
支持 CUDAExecutionProvider，FP16 输入自动利用 Tensor Core
"""
import os
import time
import logging
from typing import Tuple, List

import numpy as np
import cv2

from src.config import MODEL_CFG, SYS_CFG

logger = logging.getLogger(__name__)

# 延迟导入，未安装时给出友好提示
try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False
    ort = None


class OnnxRuntimeEngine:
    """
    ONNX Runtime GPU 推理引擎
    - 加载 .onnx 模型
    - 自动选择 CUDAExecutionProvider (GPU) 或 CPUExecutionProvider
    - FP16 模型自动利用 Volta Tensor Core (通过 cuDNN FP16 路径)
    - 接口与 TrtInferenceEngine 完全一致，main.py 无需改动
    """

    def __init__(self, model_path: str, max_batch_size: int = 1):
        if not HAS_ORT:
            raise ImportError(
                "onnxruntime 未安装。\\n"
                "Jetson 安装方法:\\n"
                "  1) pip install onnxruntime-gpu\\n"
                "  2) 若失败，从 https://elinux.org/Jetson_Zoo#ONNX_Runtime "
                "下载对应 JetPack 版本的 wheel 再 pip install"
            )

        self.model_path = model_path
        self.max_batch_size = max_batch_size

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX 模型不存在: {model_path}")

        # Session 配置
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # 执行 Provider：优先 CUDA，回退 CPU
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        provider_options = [{'device_id': 0, 'arena_extend_strategy': 'kNextPowerOfTwo'}]

        logger.info(f"加载 ONNX Runtime 模型: {model_path}")
        logger.info(f"  Providers: {providers}")
        logger.info(f"  GraphOpt: ENABLE_ALL")

        self.session = ort.InferenceSession(
            model_path,
            sess_options,
            providers=providers,
            provider_options=provider_options
        )

        # 输入信息
        inp_meta = self.session.get_inputs()[0]
        self.input_name = inp_meta.name
        self.input_shape = inp_meta.shape
        self.input_dtype = inp_meta.type  # 'tensor(float)' 或 'tensor(float16)'

        # 输出信息
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.output_shapes = [o.shape for o in self.session.get_outputs()]

        logger.info(f"  输入: {self.input_name} {self.input_shape} {self.input_dtype}")
        for i, (name, shape) in enumerate(zip(self.output_names, self.output_shapes)):
            logger.info(f"  输出[{i}]: {name} {shape}")

        # 判断模型内部是否为 FP16
        self.is_fp16_model = self.input_dtype == 'tensor(float16)'
        if self.is_fp16_model:
            logger.info("  检测到 FP16 ONNX 模型，将自动利用 Tensor Core")

    def infer(self, input_image: np.ndarray) -> List[np.ndarray]:
        """
        执行推理
        Args:
            input_image: 预处理后的图像 [N, C, H, W], float32 (由预处理保证)
        Returns:
            outputs: 模型输出列表 (与 TensorRT 格式一致)
        """
        # 确保连续内存
        if not input_image.flags['C_CONTIGUOUS']:
            input_image = np.ascontiguousarray(input_image)

        # ONNX Runtime 需要 float32 输入（即使模型内部是 FP16，ORT 会处理转换）
        if input_image.dtype != np.float32:
            input_image = input_image.astype(np.float32)

        # 执行推理
        outputs = self.session.run(self.output_names, {self.input_name: input_image})
        return outputs

    def infer_async(self, input_image: np.ndarray) -> List[np.ndarray]:
        """ONNX Runtime 不支持真正的异步，此处为接口兼容"""
        return self.infer(input_image)

    def synchronize(self):
        """ONNX Runtime 推理是同步的，无需额外同步"""
        pass

    def get_output_shapes(self) -> List[Tuple[str, Tuple, type]]:
        """返回输出形状信息（与 TensorRT 兼容格式）"""
        # 将 ONNX 类型字符串映射到 numpy dtype
        type_map = {
            'tensor(float)': np.float32,
            'tensor(float16)': np.float16,
        }
        results = []
        for o in self.session.get_outputs():
            dtype = type_map.get(o.type, np.float32)
            results.append((o.name, tuple(o.shape), dtype))
        return results

    def release(self):
        """释放资源"""
        del self.session
        logger.info("ONNX Runtime 会话已释放")


# ============================================================================
# 以下预处理函数与 tensorrt_wrapper.py 保持完全一致
# 使得 main.py 切换 backend 时无需修改预处理逻辑
# ============================================================================

def preprocess_image(image: np.ndarray,
                     target_size: Tuple[int, int] = (640, 640),
                     mean: Tuple[float, float, float] = (123.675, 116.28, 103.53),
                     std: Tuple[float, float, float] = (58.395, 57.12, 57.375)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    图像预处理 (MMPose 标准预处理)
    Args:
        image: BGR 图像 [H, W, 3]
        target_size: (width, height)
    Returns:
        (preprocessed, scale, (pad_x, pad_y))
    """
    h, w = image.shape[:2]
    tw, th = target_size

    # 1. 缩放 (保持长宽比，letterbox)
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 2. 创建画布并填充
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # 3. BGR -> RGB
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # 4. 归一化
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    canvas = (canvas.astype(np.float32) - mean_arr) / std_arr

    # 5. HWC -> CHW -> NCHW
    canvas = np.transpose(canvas, (2, 0, 1))
    canvas = np.expand_dims(canvas, axis=0)

    return canvas, scale, (pad_x, pad_y)


def preprocess_image_fast(image: np.ndarray,
                          target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    快速预处理 (使用 GPU 加速，需要 CUDA OpenCV)
    若 cv2.cuda 不可用将抛出异常，由调用方捕获并回退到 CPU 版本
    """
    h, w = image.shape[:2]
    tw, th = target_size
    scale = min(tw / w, th / h)
    new_w, new_h = int(w * scale), int(h * scale)

    # GPU 缩放
    gpu_mat = cv2.cuda_GpuMat()
    gpu_mat.upload(image)
    gpu_resized = cv2.cuda.resize(gpu_mat, (new_w, new_h))

    # 下载并后续处理
    resized = gpu_resized.download()

    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_x = (tw - new_w) // 2
    pad_y = (th - new_h) // 2
    canvas[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized

    # 归一化
    mean_arr = np.array(MODEL_CFG.mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(MODEL_CFG.std, dtype=np.float32).reshape(1, 1, 3)
    canvas = (canvas.astype(np.float32) - mean_arr) / std_arr
    canvas = np.transpose(canvas, (2, 0, 1))
    canvas = np.expand_dims(canvas, axis=0)

    return canvas, scale, (pad_x, pad_y)
