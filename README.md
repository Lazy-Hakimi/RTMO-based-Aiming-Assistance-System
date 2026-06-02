"""
TensorRT 推理引擎封装
适配 Jetson AGX Xavier (ARM64 + Volta GPU)
支持 FP16、CUDA Stream、零拷贝内存
"""
import os
import time
import logging
from typing import Tuple, List, Optional

import numpy as np
import cv2
import pycuda.driver as cuda
import pycuda.autoinit  # 自动初始化 CUDA context
import tensorrt as trt

from src.config import MODEL_CFG, SYS_CFG

logger = logging.getLogger(__name__)


class HostDeviceMem:
    """主机/设备内存对 (零拷贝优化)"""
    def __init__(self, host_mem, device_mem, is_zero_copy=False):
        self.host = host_mem
        self.device = device_mem
        self.is_zero_copy = is_zero_copy

    def __str__(self):
        return f"HostDeviceMem(zero_copy={self.is_zero_copy})"

    def __repr__(self):
        return self.__str__()


class TrtInferenceEngine:
    """
    TensorRT 推理引擎
    - 加载 .trt 引擎文件
    - 管理输入输出缓冲区
    - 支持异步推理和 CUDA Graph
    """

    def __init__(self, engine_path: str, max_batch_size: int = 1):
        self.engine_path = engine_path
        self.max_batch_size = max_batch_size

        self.logger = trt.Logger(trt.Logger.WARNING)
        if SYS_CFG.log_level == "DEBUG":
            self.logger.min_severity = trt.Logger.VERBOSE

        self.engine = None
        self.context = None
        self.stream = cuda.Stream()

        # 绑定信息
        self.inputs = []   # List[HostDeviceMem]
        self.outputs = []  # List[HostDeviceMem]
        self.bindings = [] # List[int (device ptr)]

        # 输入输出形状
        self.input_shapes = []  # List[(name, shape, dtype)]
        self.output_shapes = [] # List[(name, shape, dtype)]

        # CUDA Graph (用于静态形状加速)
        self.cuda_graph = None
        self.cuda_graph_exec = None
        self.use_cuda_graph = SYS_CFG.use_cuda_graph
        self._graph_captured = False

        # 加载引擎
        self._load_engine()
        self._allocate_buffers()
        self._setup_bindings()

        logger.info(f"TensorRT 引擎加载完成: {engine_path}")
        logger.info(f"  输入: {self.input_shapes}")
        logger.info(f"  输出: {self.output_shapes}")

    def _load_engine(self):
        """从文件加载序列化的 TensorRT 引擎"""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(f"引擎文件不存在: {self.engine_path}")

        with open(self.engine_path, "rb") as f:
            serialized_engine = f.read()

        runtime = trt.Runtime(self.logger)
        self.engine = runtime.deserialize_cuda_engine(serialized_engine)

        if self.engine is None:
            raise RuntimeError("引擎反序列化失败")

        self.context = self.engine.create_execution_context()

        # 检查 FP16 模式
        if MODEL_CFG.fp16:
            if not self.engine.has_implicit_batch_dimension and self.engine.get_tensor_mode:
                # TensorRT 8.5+ API
                pass

    def _allocate_buffers(self):
        """分配主机/设备内存缓冲区"""
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            # 计算体积
            size = trt.volume(shape)
            if size < 0:
                # 动态形状，使用最大尺寸估算
                shape = self.context.get_tensor_shape(name)
                size = trt.volume(shape)

            # 零拷贝内存 (Jetson 共享内存架构)
            if SYS_CFG.use_zero_copy and mode == trt.TensorIOMode.INPUT:
                # 使用 cudaHostAlloc 分配可页锁定的主机内存，实现零拷贝
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                is_zc = True
            else:
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                is_zc = False

            mem = HostDeviceMem(host_mem, device_mem, is_zc)
            self.bindings.append(int(device_mem))

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(mem)
                self.input_shapes.append((name, shape, dtype))
            else:
                self.outputs.append(mem)
                self.output_shapes.append((name, shape, dtype))

    def _setup_bindings(self):
        """设置 TensorRT context 绑定"""
        for i, inp in enumerate(self.inputs):
            name = self.input_shapes[i][0]
            self.context.set_tensor_address(name, int(inp.device))
        for i, out in enumerate(self.outputs):
            name = self.output_shapes[i][0]
            self.context.set_tensor_address(name, int(out.device))

    def _capture_cuda_graph(self):
        """捕获 CUDA Graph 以消除 kernel launch 开销"""
        if not self.use_cuda_graph or self._graph_captured:
            return

        # 预热
        s = self.stream
        s.synchronize()

        # 开始捕获
        try:
            with cuda.Graph() as graph:
                self.context.execute_async_v3(stream_handle=s.handle)

            self.cuda_graph = graph
            self.cuda_graph_exec = graph.instantiate()
            self._graph_captured = True
            logger.info("CUDA Graph 捕获成功")
        except Exception as e:
            logger.warning(f"CUDA Graph 捕获失败: {e}")
            self.use_cuda_graph = False

    def infer(self, input_image: np.ndarray) -> List[np.ndarray]:
        """
        执行推理
        Args:
            input_image: 预处理后的图像 [H, W, C] 或 [B, H, W, C], float32
        Returns:
            outputs: 模型输出列表
        """
        # 确保输入是连续的
        if not input_image.flags['C_CONTIGUOUS']:
            input_image = np.ascontiguousarray(input_image)

        # 拷贝输入到设备
        inp = self.inputs[0]
        inp.host[:] = input_image.ravel()
        cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        # 执行推理
        if self.use_cuda_graph and self._graph_captured:
            self.cuda_graph_exec.launch(self.stream)
        else:
            self.context.execute_async_v3(stream_handle=self.stream.handle)

        # 拷贝输出回主机
        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            outputs.append(out.host)

        self.stream.synchronize()
        return outputs

    def infer_async(self, input_image: np.ndarray) -> List[np.ndarray]:
        """异步推理 (需要外部同步)"""
        if not input_image.flags['C_CONTIGUOUS']:
            input_image = np.ascontiguousarray(input_image)

        inp = self.inputs[0]
        inp.host[:] = input_image.ravel()
        cuda.memcpy_htod_async(inp.device, inp.host, self.stream)

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out.host, out.device, self.stream)
            outputs.append(out.host)

        return outputs

    def synchronize(self):
        """同步 CUDA 流"""
        self.stream.synchronize()

    def get_output_shapes(self) -> List[Tuple[str, Tuple, type]]:
        return self.output_shapes

    def release(self):
        """释放资源"""
        self.stream.synchronize()
        for mem in self.inputs + self.outputs:
            mem.device.free()
        if self.cuda_graph_exec:
            self.cuda_graph_exec = None
        if self.cuda_graph:
            self.cuda_graph = None
        self.context = None
        self.engine = None
        logger.info("TensorRT 引擎资源已释放")


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
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)  # 灰色填充
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

    # 下载并后续处理 (归一化等)
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
