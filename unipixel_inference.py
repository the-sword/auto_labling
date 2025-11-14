"""
UniPixel推理模块 - 替换Grounding DINO + SAM
基于 UniPixel: Unified Object Referring and Segmentation
Repository: https://github.com/PolyU-ChenLab/UniPixel
"""

import os
import sys
from typing import List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

# 添加UniPixel路径到sys.path
UNIPIXEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "UniPixel-main")
if UNIPIXEL_PATH not in sys.path:
    sys.path.insert(0, UNIPIXEL_PATH)

try:
    from unipixel.model.builder import build_model
    from unipixel.dataset.utils import process_vision_info
    from unipixel.utils.transforms import get_sam2_transform
    UNIPIXEL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: UniPixel not available: {e}")
    UNIPIXEL_AVAILABLE = False


@dataclass
class BoundingBox:
    """边界框数据类"""
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    """检测结果数据类"""
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None


class UniPixelInference:
    """UniPixel推理封装类"""
    
    def __init__(self, model_path: str = "PolyU-ChenLab/UniPixel-3B", 
                 device: str = "auto",
                 dtype: str = "bfloat16"):
        """
        初始化UniPixel模型
        
        Args:
            model_path: 模型路径或HuggingFace模型ID
            device: 设备 ('auto', 'cuda', 'cpu')
            dtype: 数据类型 ('bfloat16', 'float16', 'float32')
        """
        if not UNIPIXEL_AVAILABLE:
            raise ImportError("UniPixel dependencies not available. Please install requirements.")
        
        print(f"Loading UniPixel model from {model_path}...")
        self.model, self.processor = build_model(
            model_path, 
            device=device, 
            dtype=dtype
        )
        self.device = next(self.model.parameters()).device
        self.sam2_transform = get_sam2_transform(self.model.config.sam2_image_size)
        print(f"UniPixel model loaded on device: {self.device}")
    
    def _create_prompt(self, labels: List[str]) -> str:
        """
        从标签列表创建提示词
        
        Args:
            labels: 标签列表，如 ['cat', 'dog']
            
        Returns:
            提示词字符串
        """
        if len(labels) == 1:
            return f"Please segment the {labels[0]}."
        else:
            # 多标签情况，生成多个对象的prompt
            objects = ", ".join(labels)
            return f"Please segment all instances of the following objects: {objects}. Provide segmentation masks for each object."
    
    def _extract_masks_from_model(self) -> List[np.ndarray]:
        """
        从模型中提取分割mask
        
        Returns:
            mask数组列表
        """
        if not hasattr(self.model, 'seg') or len(self.model.seg) == 0:
            return []
        
        masks = []
        for seg_item in self.model.seg:
            # seg_item可能是tensor或numpy数组
            if isinstance(seg_item, torch.Tensor):
                mask = seg_item.cpu().numpy()
            else:
                mask = np.array(seg_item)
            
            # 确保mask是2D布尔数组
            if mask.ndim > 2:
                mask = mask.squeeze()
            
            masks.append(mask.astype(bool))
        
        return masks
    
    def _create_detection_results(self, 
                                   masks: List[np.ndarray], 
                                   labels: List[str],
                                   threshold: float = 0.3) -> List[DetectionResult]:
        """
        从masks创建DetectionResult对象
        
        Args:
            masks: mask数组列表
            labels: 标签列表
            threshold: 置信度阈值（UniPixel为端到端模型，默认给高分）
            
        Returns:
            DetectionResult列表
        """
        results = []
        
        for i, mask in enumerate(masks):
            # 计算边界框
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            
            if not rows.any() or not cols.any():
                continue
            
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
            
            box = BoundingBox(
                xmin=int(xmin),
                ymin=int(ymin),
                xmax=int(xmax),
                ymax=int(ymax)
            )
            
            # 确定标签（循环使用labels列表）
            label = labels[i % len(labels)]
            if label.endswith('.'):
                label = label[:-1]
            
            result = DetectionResult(
                score=0.95,  # UniPixel是端到端模型，给予高置信度
                label=label,
                box=box,
                mask=mask
            )
            
            results.append(result)
        
        return results
    
    def segment(self, 
                image: Image.Image, 
                labels: List[str], 
                threshold: float = 0.3) -> List[DetectionResult]:
        """
        对单张图片进行分割
        
        Args:
            image: PIL图像
            labels: 目标标签列表
            threshold: 置信度阈值
            
        Returns:
            DetectionResult列表
        """
        # 创建提示词
        prompt = self._create_prompt(labels)
        
        # 准备输入数据
        image_np = np.array(image.convert("RGB"))
        
        # 构建messages
        messages = [{
            'role': 'user',
            'content': [
                {
                    'type': 'image',
                    'image': image,
                    'min_pixels': 128 * 28 * 28,
                    'max_pixels': 256 * 28 * 28
                },
                {
                    'type': 'text',
                    'text': prompt
                }
            ]
        }]
        
        # 处理输入
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, return_video_kwargs=True)
        data = self.processor(text=[text], images=images, videos=videos, return_tensors='pt', **kwargs)
        
        # 添加SAM2所需的frame数据
        frames = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        frames = frames.unsqueeze(0)  # [1, 3, H, W]
        
        data['frames'] = [self.sam2_transform(frames.permute(1, 2, 3, 0).numpy()).to(self.model.sam2.dtype)]
        data['frame_size'] = [image_np.shape[:2]]
        
        # 清空之前的分割结果
        if hasattr(self.model, 'seg'):
            self.model.seg = []
        
        # 推理
        with torch.no_grad():
            output_ids = self.model.generate(
                **data.to(self.device),
                do_sample=False,
                temperature=None,
                top_k=None,
                top_p=None,
                repetition_penalty=None,
                max_new_tokens=512
            )
        
        # 解码响应
        assert data.input_ids.size(0) == output_ids.size(0) == 1
        output_ids = output_ids[0, data.input_ids.size(1):]
        
        if output_ids[-1] == self.processor.tokenizer.eos_token_id:
            output_ids = output_ids[:-1]
        
        response = self.processor.decode(output_ids, clean_up_tokenization_spaces=False)
        print(f"UniPixel Response: {response}")
        
        # 提取masks
        masks = self._extract_masks_from_model()
        
        # 创建DetectionResult
        results = self._create_detection_results(masks, labels, threshold)
        
        return results
    
    def segment_batch(self,
                     images: List[Image.Image],
                     labels: List[str],
                     threshold: float = 0.3) -> List[List[DetectionResult]]:
        """
        批量处理多张图片
        
        Args:
            images: PIL图像列表
            labels: 目标标签列表
            threshold: 置信度阈值
            
        Returns:
            每张图片的DetectionResult列表
        """
        # UniPixel当前实现为逐张处理
        # TODO: 探索真正的批处理优化
        results = []
        for image in images:
            result = self.segment(image, labels, threshold)
            results.append(result)
        
        return results


# 全局模型实例
_global_unipixel_model: Optional[UniPixelInference] = None


def get_unipixel_model(model_path: str = "PolyU-ChenLab/UniPixel-3B") -> UniPixelInference:
    """
    获取或创建全局UniPixel模型实例
    
    Args:
        model_path: 模型路径
        
    Returns:
        UniPixelInference实例
    """
    global _global_unipixel_model
    
    if _global_unipixel_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _global_unipixel_model = UniPixelInference(
            model_path=model_path,
            device=device,
            dtype="bfloat16" if device == "cuda" else "float32"
        )
    
    return _global_unipixel_model
