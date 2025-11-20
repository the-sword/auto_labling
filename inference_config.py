"""
推理引擎配置
"""

import os

INFERENCE_ENGINE = os.getenv('INFERENCE_ENGINE', 'unipixel')

# UniPixel模型配置
UNIPIXEL_MODEL_PATH = os.getenv('UNIPIXEL_MODEL_PATH', 'PolyU-ChenLab/UniPixel-3B')

# Grounding DINO + SAM配置（用于向后兼容）
GROUNDING_DINO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/grounding-dino-tiny")
SAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/sam-vit-base")

# SAM3 HTTP 推理服务配置
SAM3_HTTP_URL = os.getenv('SAM3_HTTP_URL', 'http://localhost:6666/sam3/segment')

def get_inference_engine():
    """获取当前配置的推理引擎名称"""
    return INFERENCE_ENGINE

def set_inference_engine(engine: str):
    """
    设置推理引擎

    Args:
        engine: 'grounding_dino_sam' 或 'unipixel' 或 'sam3_http'
    """
    global INFERENCE_ENGINE
    if engine not in ['grounding_dino_sam', 'unipixel', 'sam3_http']:
        raise ValueError(f"Invalid engine: {engine}. Must be 'grounding_dino_sam', 'unipixel' or 'sam3_http'")
    INFERENCE_ENGINE = engine
    print(f"Inference engine set to: {engine}")
