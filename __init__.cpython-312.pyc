#!/usr/bin/env python3
"""
从 MMPose 导出 RTMO 模型为 ONNX

步骤：
1. 安装 MMPose + MMDeploy
2. 下载 RTMO-l 预训练权重
3. 修改 MMDeploy 后处理代码 (剔除 NMS)
4. 导出 ONNX

注意：此脚本需要在 x86 开发机上运行 (Jetson 上安装 MMPose 较困难)
导出后的 ONNX 再转换为 TensorRT 引擎部署到 Jetson
"""
import os
import sys
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_onnx(config_path: str, checkpoint_path: str, output_dir: str, 
                opset_version: int = 11, input_shape: tuple = (640, 640)):
    """
    导出 RTMO ONNX 模型
    """
    try:
        import torch
        from mmdeploy.apis import export_model
        from mmdeploy.utils import get_root_logger
    except ImportError as e:
        logger.error(f"缺少依赖: {e}")
        logger.error("请安装: pip install mmpose mmdeploy onnx")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # 导出配置
    deploy_config = "configs/mmpose/pose-detection_rtmo_onnxruntime_dynamic.py"

    # 如果找不到配置文件，使用内联配置
    if not os.path.exists(deploy_config):
        logger.warning(f"找不到配置文件: {deploy_config}")
        logger.info("使用内联配置...")
        deploy_config = create_inline_deploy_config(output_dir)

    logger.info(f"配置: {config_path}")
    logger.info(f"权重: {checkpoint_path}")
    logger.info(f"输出: {output_dir}")

    # 执行导出
    export_model(
        deploy_cfg=deploy_config,
        model_cfg=config_path,
        model_checkpoint=checkpoint_path,
        input_shape=input_shape,
        backend="ONNXRuntime",
        output_path=output_dir,
        device="cuda:0"
    )

    logger.info("ONNX 导出完成")

    # 简化 ONNX (使用 onnxsim)
    try:
        import onnx
        from onnxsim import simplify

        onnx_path = os.path.join(output_dir, "end2end.onnx")
        if os.path.exists(onnx_path):
            logger.info("简化 ONNX 模型...")
            model = onnx.load(onnx_path)
            model_simp, check = simplify(model)
            if check:
                onnx.save(model_simp, onnx_path)
                logger.info("ONNX 简化完成")
    except Exception as e:
        logger.warning(f"ONNX 简化失败: {e}")


def create_inline_deploy_config(output_dir: str) -> str:
    """创建内联部署配置"""
    config_content = """
backend_config = dict(type='onnxruntime')
codebase_config = dict(type='mmpose', task='PoseDetection')

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['dets', 'keypoints'],
    input_shape=None,
    optimize=True,
    dynamic_axes={
        'input': {
            0: 'batch',
            2: 'height',
            3: 'width'
        },
        'dets': {
            0: 'batch',
            1: 'num_dets'
        },
        'keypoints': {
            0: 'batch',
            1: 'num_dets'
        }
    }
)
"""
    config_path = os.path.join(output_dir, "deploy_config.py")
    with open(config_path, "w") as f:
        f.write(config_content)
    return config_path


def modify_mmdeploy_for_clean_export():
    """
    修改 MMDeploy 源码，剔除 NMS 后处理
    这是获得干净 ONNX 的关键步骤
    """
    import mmdeploy
    mmdeploy_path = os.path.dirname(mmdeploy.__file__)
    rtmo_head_path = os.path.join(
        mmdeploy_path, "codebase/mmpose/models/heads/rtmo_head.py"
    )

    if not os.path.exists(rtmo_head_path):
        logger.warning(f"找不到文件: {rtmo_head_path}")
        return

    logger.info(f"修改文件: {rtmo_head_path}")
    logger.info("请手动注释掉 NMS 相关代码，保留原始输出")
    logger.info("参考修改:")
    logger.info("""
    # 原始代码 (约 68 行):
    # _, _, nms_indices = multiclass_nms(...)
    # dets = dets[batch_inds, nms_indices, ...]
    # pose_vecs = flatten_pose_vecs[batch_inds, nms_indices, ...]

    # 修改为:
    dets = torch.cat([bboxes, scores], dim=2)
    pose_vecs = flatten_pose_vecs
    kpt_vis = flatten_kpt_vis
    grids = self.flatten_priors
    """)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True,
                       help="MMPose 配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="预训练权重路径")
    parser.add_argument("--output", type=str, default="models",
                       help="输出目录")
    parser.add_argument("--opset", type=int, default=11,
                       help="ONNX opset 版本")
    parser.add_argument("--modify-mmdeploy", action="store_true",
                       help="显示 MMDeploy 修改说明")

    args = parser.parse_args()

    if args.modify_mmdeploy:
        modify_mmdeploy_for_clean_export()
        return

    export_onnx(args.config, args.checkpoint, args.output, args.opset)


if __name__ == "__main__":
    main()
