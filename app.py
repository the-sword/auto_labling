import os
import base64
import io
import json
import random
import signal
import sys
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
object_detector = None
segmentator = None
processor = None

@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]

@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> 'DetectionResult':
        return cls(score=detection_dict['score'],
                   label=detection_dict['label'],
                   box=BoundingBox(xmin=detection_dict['box']['xmin'],
                                   ymin=detection_dict['box']['ymin'],
                                   xmax=detection_dict['box']['xmax'],
                                   ymax=detection_dict['box']['ymax']))

def simplify_polygon(points: List[List[int]], epsilon: float = 2.0, collinear_eps: float = 1.0) -> List[List[int]]:
    """简化多边形点集：
    - 使用 Ramer–Douglas–Peucker（approxPolyDP）按像素误差 epsilon 简化
    - 进一步删除共线点（相邻三点构成的三角形面积 < collinear_eps）
    """
    if not points or len(points) < 3:
        return points

    # 先用 approxPolyDP（RDP）
    cnt = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(cnt, epsilon=epsilon, closed=True)
    poly = approx.reshape(-1, 2).tolist()

    # 再去除共线点
    def area2(a, b, c):
        return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

    if len(poly) > 3:
        filtered = []
        n = len(poly)
        for i in range(n):
            prev = poly[(i - 1) % n]
            cur = poly[i]
            nxt = poly[(i + 1) % n]
            # 面积为0意味着完全共线；允许一个小阈值
            if area2(prev, cur, nxt) <= collinear_eps:
                continue
            filtered.append(cur)
        # 保证至少三点
        if len(filtered) >= 3:
            poly = filtered
    return poly

def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0, collinear_eps: float = 1.0) -> List[List[int]]:
    """将mask转换为简化后的多边形坐标"""
    if mask is None:
        return []

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)
    polygon = largest_contour.reshape(-1, 2).tolist()
    polygon = simplify_polygon(polygon, epsilon=epsilon, collinear_eps=collinear_eps)
    return polygon

def get_boxes(results: List[DetectionResult]) -> List[List[float]]:
    """提取边界框坐标"""
    boxes = []
    for result in results:
        xyxy = result.box.xyxy
        boxes.append(xyxy)
    return boxes

def bbox_iou(a: BoundingBox, b: BoundingBox) -> float:
    """计算两个边界框的IoU (Intersection over Union)"""
    xA = max(a.xmin, b.xmin)
    yA = max(a.ymin, b.ymin)
    xB = min(a.xmax, b.xmax)
    yB = min(a.ymax, b.ymax)

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_a = max(0, a.xmax - a.xmin) * max(0, a.ymax - a.ymin)
    area_b = max(0, b.xmax - b.xmin) * max(0, b.ymax - b.ymin)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union

def nms_detections(detections: List[DetectionResult], iou_threshold: float = 0.5) -> List[DetectionResult]:
    """对检测结果执行NMS，保留重叠度高的框中的最高分一项，避免一个物体被多个标签覆盖"""
    if not detections:
        return detections

    # 按分数从高到低排序
    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[DetectionResult] = []

    for det in sorted_dets:
        overlap = False
        for kept_det in kept:
            iou = bbox_iou(det.box, kept_det.box)
            if iou >= iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)

    return kept

def mask_iou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    """计算两个二值mask的IoU。若任一为空或形状不一致，返回0。"""
    if mask_a is None or mask_b is None:
        return 0.0
    if mask_a.shape != mask_b.shape:
        return 0.0
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)

def mask_level_nms(detections: List[DetectionResult], mask_iou_threshold: float = 0.5) -> List[DetectionResult]:
    """
    基于mask的跨类别NMS：
    - 先按score从高到低排序
    - 若一个候选与已保留候选的mask IoU>=阈值，则丢弃（认为是同一物体的重复标签），保留分数更高者
    - 这样当同一物体被打上多个标签时，只保留最高分的那个标签
    """
    if not detections:
        return detections

    sorted_dets = sorted(detections, key=lambda d: d.score, reverse=True)
    kept: List[DetectionResult] = []
    for det in sorted_dets:
        duplicate = False
        for k in kept:
            iou = mask_iou(det.mask, k.mask)
            if iou >= mask_iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept

def refine_masks(masks: torch.BoolTensor, polygon_refinement: bool = False,
                 poly_simplify_eps: float = 2.0, poly_collinear_eps: float = 1.0) -> List[np.ndarray]:
    """优化mask"""
    masks = masks.cpu().float()
    masks = masks.permute(0, 2, 3, 1)
    masks = masks.mean(axis=-1)
    masks = (masks > 0).int()
    masks = masks.numpy().astype(np.uint8)
    masks = list(masks)

    if polygon_refinement:
        for idx, mask in enumerate(masks):
            shape = mask.shape
            polygon = mask_to_polygon(mask, epsilon=poly_simplify_eps, collinear_eps=poly_collinear_eps)
            if polygon:
                mask = polygon_to_mask(polygon, shape)
                masks[idx] = mask

    return masks

def polygon_to_mask(polygon: List[Tuple[int, int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """将多边形转换为mask"""
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], color=(255,))
    return mask

def load_image(image_data: bytes) -> Image.Image:
    """从字节数据加载图像"""
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    return image

def detect(image: Image.Image, labels: List[str], threshold: float = 0.3) -> List[DetectionResult]:
    """使用Grounding DINO检测对象"""
    global object_detector

    if object_detector is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/grounding-dino-tiny")
        object_detector = pipeline(
            model=model_path,
            task="zero-shot-object-detection",
            device=device,
            local_files_only=True
        )

    labels = [label if label.endswith(".") else label+"." for label in labels]
    results = object_detector(image, candidate_labels=labels, threshold=threshold)
    results = [DetectionResult.from_dict(result) for result in results]

    # 取消框级NMS，保留所有候选，后续在mask层级进行冲突消解
    return results

def segment(image: Image.Image, detection_results: List[DetectionResult], polygon_refinement: bool = False,
            mask_iou_threshold: float = 0.5,
            poly_simplify_eps: float = 2.0,
            poly_collinear_eps: float = 1.0) -> List[DetectionResult]:
    """使用SAM生成分割mask"""
    global segmentator, processor

    if segmentator is None or processor is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device set to use {device}")
        sam_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/sam-vit-base")
        segmentator = AutoModelForMaskGeneration.from_pretrained(sam_path, local_files_only=True).to(device)
        processor = AutoProcessor.from_pretrained(sam_path, local_files_only=True)

    if not detection_results:
        return detection_results

    boxes = get_boxes(detection_results)
    # print(f"boxes: {boxes}")
    inputs = processor(images=image, input_boxes=[boxes], return_tensors="pt").to(segmentator.device)
    # print(f"inputs: {inputs}")
    with torch.no_grad():
        outputs = segmentator(**inputs)

    masks = processor.post_process_masks(
        masks=outputs.pred_masks,
        original_sizes=inputs.original_sizes,
        reshaped_input_sizes=inputs.reshaped_input_sizes
    )[0]

    masks = refine_masks(masks, polygon_refinement, poly_simplify_eps=poly_simplify_eps, poly_collinear_eps=poly_collinear_eps)

    for detection_result, mask in zip(detection_results, masks):
        detection_result.mask = mask

    # 基于mask进行跨类别去重，避免同一物体被多个标签覆盖
    detection_results = mask_level_nms(detection_results, mask_iou_threshold=mask_iou_threshold)

    return detection_results

def grounded_segmentation(image_data: bytes, labels: List[str], threshold: float = 0.3,
                          polygon_refinement: bool = False, mask_iou_threshold: float = 0.5,
                          poly_simplify_eps: float = 2.0, poly_collinear_eps: float = 1.0) -> Tuple[np.ndarray, List[DetectionResult]]:
    """执行完整的grounded segmentation流程"""
    image = load_image(image_data)
    detections = detect(image, labels, threshold)
    detections = segment(image, detections, polygon_refinement, mask_iou_threshold,
                         poly_simplify_eps=poly_simplify_eps, poly_collinear_eps=poly_collinear_eps)
    # print(f"detections: {detections}")
    return np.array(image), detections

def detection_result_to_dict(detection: DetectionResult, poly_simplify_eps: float = 2.0, poly_collinear_eps: float = 1.0) -> Dict:
    """将DetectionResult转换为可序列化的字典"""
    result = {
        'score': float(detection.score),
        'label': detection.label,
        'box': {
            'xmin': detection.box.xmin,
            'ymin': detection.box.ymin,
            'xmax': detection.box.xmax,
            'ymax': detection.box.ymax
        }
    }

    # 标记手动标注来源（若存在）
    if hasattr(detection, 'is_manual') and getattr(detection, 'is_manual'):
        result['is_manual'] = True

    if detection.mask is not None:
        # 将mask转换为base64编码
        mask_pil = Image.fromarray(detection.mask.astype(np.uint8) * 255)
        buffer = io.BytesIO()
        mask_pil.save(buffer, format='PNG')
        mask_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        result['mask'] = mask_base64

        # 添加多边形坐标
        polygon = mask_to_polygon(detection.mask, epsilon=poly_simplify_eps, collinear_eps=poly_collinear_eps)
        result['polygon'] = polygon

    return result

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/segment', methods=['POST'])
def segment_api():
    """分割API端点"""
    try:
        # 获取请求数据
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        labels = data['labels']
        threshold = data.get('threshold', 0.3)
        polygon_refinement = data.get('polygon_refinement', True)
        mask_iou_threshold = float(data.get('mask_iou_threshold', 0.5))
        manual_annotations = data.get('manual_annotations', []) or []

        # 读取多边形简化参数
        try:
            poly_simplify_eps = float(data.get('polygon_simplify_epsilon', 2.0))
        except Exception:
            poly_simplify_eps = 2.0
        try:
            poly_collinear_eps = float(data.get('polygon_collinear_epsilon', 1.0))
        except Exception:
            poly_collinear_eps = 1.0

        # 执行分割（自动检测+分割）
        image_array, detections = grounded_segmentation(
            image_data, labels, threshold, polygon_refinement, mask_iou_threshold,
            poly_simplify_eps=poly_simplify_eps, poly_collinear_eps=poly_collinear_eps
        )

        # 将手动标注转换为 DetectionResult，并合并
        try:
            h, w = int(Image.open(io.BytesIO(image_data)).height), int(Image.open(io.BytesIO(image_data)).width)
        except Exception:
            # 回退：从已生成的image_array推断
            h, w = int(image_array.shape[0]), int(image_array.shape[1])

        for ann in manual_annotations:
            polygon = ann.get('polygon') or []
            if not isinstance(polygon, list) or len(polygon) < 3:
                continue
            label = str(ann.get('label', 'manual'))
            try:
                score = float(ann.get('score', 1.0))
            except Exception:
                score = 1.0
            # 计算/校验bbox
            box_dict = ann.get('box') or {}
            if not all(k in box_dict for k in ('xmin', 'ymin', 'xmax', 'ymax')):
                xs = [int(p[0]) for p in polygon]
                ys = [int(p[1]) for p in polygon]
                box_dict = {
                    'xmin': int(max(0, min(xs))),
                    'ymin': int(max(0, min(ys))),
                    'xmax': int(min(w - 1, max(xs))),
                    'ymax': int(min(h - 1, max(ys)))
                }

            box = BoundingBox(
                xmin=int(box_dict['xmin']),
                ymin=int(box_dict['ymin']),
                xmax=int(box_dict['xmax']),
                ymax=int(box_dict['ymax'])
            )

            # 生成mask并归一化为0/1
            mask = polygon_to_mask([(int(p[0]), int(p[1])) for p in polygon], (h, w))
            mask = (mask > 0).astype(np.uint8)

            det = DetectionResult(score=score, label=label, box=box, mask=mask)
            # 动态标记为手动
            setattr(det, 'is_manual', True)
            detections.append(det)

        # 融合后在mask层级去重（避免多标签覆盖同一物体）
        detections = mask_level_nms(detections, mask_iou_threshold=mask_iou_threshold)

        # 转换结果为JSON格式
        results = [detection_result_to_dict(detection, poly_simplify_eps, poly_collinear_eps) for detection in detections]

        # 将原图转换为base64
        image_pil = Image.fromarray(image_array)
        buffer = io.BytesIO()
        image_pil.save(buffer, format='PNG')
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return jsonify({
            'success': True,
            'image': image_base64,
            'detections': results
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查端点"""
    return jsonify({'status': 'healthy'})

def signal_handler(sig, frame):
    """处理终止信号，确保应用程序正确关闭"""
    print('\n正在关闭应用程序...')
    # 释放资源
    global object_detector, segmentator, processor
    object_detector = None
    segmentator = None
    processor = None
    # 强制清理GPU内存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    sys.exit(0)

if __name__ == '__main__':
    # 注册信号处理函数
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print('\n接收到键盘中断，正在关闭...')
        # 确保资源被释放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
