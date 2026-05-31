#!/usr/bin/env python3
"""
ONNX -> TensorRT 引擎转换脚本
在 Jetson AGX Xavier 上运行

使用方法：
    python3 scripts/onnx2trt.py --onnx models/rtmo_l_640x640.onnx \
                                --output models/rtmo_l_640x640.trt \
                                --fp16 --workspace 2048
"""
import os
import sys
import argparse
import logging

import tensorrt as trt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class TensorRTBuilder:
    """TensorRT 引擎构建器"""

    def __init__(self, workspace_mb: int = 2048, fp16: bool = True):
        self.logger = trt.Logger(trt.Logger.INFO)
        self.workspace_mb = workspace_mb
        self.fp16 = fp16

    def build_engine(self, onnx_path: str, output_path: str, 
                     max_batch_size: int = 1, 
                     input_shape: tuple = (1, 3, 640, 640)) -> bool:
        """
        构建 TensorRT 引擎
        Args:
            onnx_path: ONNX 模型路径
            output_path: 输出引擎路径
            max_batch_size: 最大 batch size
            input_shape: 输入形状 (N, C, H, W)
        """
        logger.info(f"构建 TensorRT 引擎")
        logger.info(f"  ONNX: {onnx_path}")
        logger.info(f"  输出: {output_path}")
        logger.info(f"  FP16: {self.fp16}")
        logger.info(f"  Workspace: {self.workspace_mb}MB")

        builder = trt.Builder(self.logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, self.logger)

        # 解析 ONNX
        logger.info("解析 ONNX 模型...")
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                logger.error("ONNX 解析失败:")
                for i in range(parser.num_errors):
                    logger.error(f"  {parser.get_error(i)}")
                return False

        # 配置 builder
        config = builder.create_builder_config()
        config.max_workspace_size = self.workspace_mb * (1 << 20)  # MB -> bytes

        # 设置 FP16
        if self.fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用 FP16 模式")
            else:
                logger.warning("平台不支持 FP16")

        # 设置 INT8 (可选)
        # if builder.platform_has_fast_int8:
        #     config.set_flag(trt.BuilderFlag.INT8)

        # 设置输入形状
        input_tensor = network.get_input(0)
        input_tensor.shape = input_shape

        # 构建引擎
        logger.info("构建引擎 (可能需要几分钟)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            logger.error("引擎构建失败")
            return False

        # 序列化保存
        logger.info("序列化引擎...")
        with open(output_path, "wb") as f:
            f.write(engine.serialize())

        logger.info(f"引擎已保存: {output_path}")
        logger.info(f"  引擎大小: {os.path.getsize(output_path) / (1024*1024):.1f} MB")

        # 打印引擎信息
        self._print_engine_info(engine)

        return True

    def _print_engine_info(self, engine):
        """打印引擎信息"""
        logger.info("引擎信息:")
        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            mode = engine.get_tensor_mode(name)
            shape = engine.get_tensor_shape(name)
            dtype = engine.get_tensor_dtype(name)
            mode_str = "INPUT" if mode == trt.TensorIOMode.INPUT else "OUTPUT"
            logger.info(f"  [{mode_str}] {name}: {shape} {dtype}")


def main():
    parser = argparse.ArgumentParser(description="ONNX to TensorRT Engine Converter")
    parser.add_argument("--onnx", type=str, required=True,
                       help="ONNX 模型路径")
    parser.add_argument("--output", type=str, required=True,
                       help="输出引擎路径")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="启用 FP16")
    parser.add_argument("--no-fp16", action="store_true",
                       help="禁用 FP16")
    parser.add_argument("--workspace", type=int, default=2048,
                       help="工作空间大小 (MB)")
    parser.add_argument("--batch", type=int, default=1,
                       help="最大 batch size")
    parser.add_argument("--height", type=int, default=640,
                       help="输入高度")
    parser.add_argument("--width", type=int, default=640,
                       help="输入宽度")

    args = parser.parse_args()

    fp16 = args.fp16 and not args.no_fp16

    builder = TensorRTBuilder(workspace_mb=args.workspace, fp16=fp16)
    input_shape = (args.batch, 3, args.height, args.width)

    success = builder.build_engine(
        args.onnx, args.output, 
        max_batch_size=args.batch,
        input_shape=input_shape
    )

    if not success:
        sys.exit(1)

    logger.info("转换完成!")


if __name__ == "__main__":
    main()
