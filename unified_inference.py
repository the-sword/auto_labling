"""统一推理接口 - 支持多种推理后端"""

from typing import List, Any
from PIL import Image

import inference_config


class UnifiedInferenceEngine:
    """统一推理引擎 - 抽象不同的推理后端"""

    def __init__(self):
        self.engine_type = inference_config.get_inference_engine()
        self._backend = None
        print(f"Initializing UnifiedInferenceEngine with backend: {self.engine_type}")

    def _get_backend(self):
        """懒加载推理后端"""
        if self._backend is not None:
            return self._backend

        if self.engine_type == 'sam3_http':
            from sam3_http_inference import Sam3HTTPInference
            self._backend = Sam3HTTPInference(inference_config.SAM3_HTTP_URL)
        elif self.engine_type == 'grounding_dino_sam':
            # 使用原有的Grounding DINO + SAM
            from app import detect, segment as sam_segment

            class GroundingDINOSAMBackend:
                """Grounding DINO + SAM后端适配器"""
                def segment(self, image: Image.Image, labels: List[str], threshold: float = 0.3):
                    # 使用原有的detect和segment函数
                    detections = detect(image, labels, threshold)
                    results = sam_segment(image, detections, polygon_refinement=True)
                    return results

                def segment_batch(self, images: List[Image.Image], labels: List[str], threshold: float = 0.3):
                    from app import grounded_segmentation_batch
                    import io
                    # 转换图片为bytes
                    image_bytes = []
                    for img in images:
                        buf = io.BytesIO()
                        img.save(buf, format='PNG')
                        image_bytes.append(buf.getvalue())

                    _, results = grounded_segmentation_batch(
                        image_bytes, labels, threshold,
                        polygon_refinement=True,
                        mask_iou_threshold=0.5
                    )
                    return results

            self._backend = GroundingDINOSAMBackend()
        else:
            raise ValueError(f"Unknown inference engine: {self.engine_type}")

        return self._backend

    def segment(self, image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[Any]:
        """
        对单张图片进行分割

        Args:
            image: PIL图像
            labels: 目标标签列表
            threshold: 置信度阈值

        Returns:
            DetectionResult列表
        """
        backend = self._get_backend()
        return backend.segment(image, labels, threshold)

    def segment_batch(self, images: List[Image.Image], labels: List[str], threshold: float = 0.3) -> List[List[Any]]:
        """
        批量处理多张图片

        Args:
            images: PIL图像列表
            labels: 目标标签列表
            threshold: 置信度阈值

        Returns:
            每张图片的DetectionResult列表
        """
        backend = self._get_backend()
        return backend.segment_batch(images, labels, threshold)

    def segment_by_points(self, image: Image.Image, points: List[List[int]], point_labels: List[int], threshold: float = 0.3) -> List[Any]:
        """
        使用点选提示进行分割

        Args:
            image: PIL图像
            points: 点坐标列表 [[x1, y1], [x2, y2], ...]
            point_labels: 点标签列表 [1, 0, ...] (1=前景点, 0=背景点)
            threshold: 置信度阈值

        Returns:
            DetectionResult列表
        """
        backend = self._get_backend()
        if hasattr(backend, 'segment_by_points'):
            return backend.segment_by_points(image, points, point_labels, threshold)
        else:
            raise NotImplementedError(f"Backend {self.engine_type} does not support point-based segmentation")

    def reload(self):
        """重新加载后端（用于切换引擎）"""
        self._backend = None
        self.engine_type = inference_config.get_inference_engine()
        print(f"Reloading inference backend: {self.engine_type}")


# 全局统一推理引擎实例
_global_engine = None


def get_unified_engine() -> UnifiedInferenceEngine:
    """获取全局统一推理引擎实例"""
    global _global_engine
    if _global_engine is None:
        _global_engine = UnifiedInferenceEngine()
    return _global_engine
