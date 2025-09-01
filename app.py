import os
import base64
import io
import json
import random
import time
import signal
import sys
import shutil
from dataclasses import dataclass, asdict
from typing import Any, List, Dict, Optional, Union, Tuple

import cv2
import torch
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, Response, stream_with_context
from flask_cors import CORS
from transformers import AutoModelForMaskGeneration, AutoProcessor, pipeline

# 导入文件工具模块
import file_utils
import crawler

# ================= augmentation =================
from threading import Event

# 全局停止事件
AUGMENT_STOP_EVENT = Event()

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
object_detector = None
segmentator = None
processor = None

# 使用file_utils中的文件夹配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 获取文件夹路径
folders = file_utils.get_current_folders()
UPLOAD_FOLDER = folders['upload_folder']
RESULTS_FOLDER = folders['results_folder']

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
    return file_utils.simplify_polygon(points, epsilon, collinear_eps)

def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0, collinear_eps: float = 1.0) -> List[List[int]]:
    """将mask转换为简化后的多边形坐标"""
    return file_utils.mask_to_polygon(mask, epsilon, collinear_eps)

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
            polygon = mask_to_polygon(mask, epsilon=0.8, collinear_eps=1.0)
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

def load_image_from_path(path: str) -> bytes:
    """从磁盘路径读取文件字节（限制在UPLOAD_FOLDER内）"""
    abs_path = os.path.abspath(os.path.join(UPLOAD_FOLDER, path)) if not os.path.isabs(path) else os.path.abspath(path)
    # 安全限制：必须在UPLOAD_FOLDER目录内
    if not abs_path.startswith(os.path.abspath(UPLOAD_FOLDER)):
        raise ValueError('Invalid image_path')
    with open(abs_path, 'rb') as f:
        return f.read()

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
        image_b64 = data.get('image')
        image_path = data.get('image_path')  # 相对或绝对路径，首选相对 uploads 下
        if image_b64:
            image_data = base64.b64decode(image_b64.split(',')[1])
        elif image_path:
            image_data = load_image_from_path(image_path)
        else:
            return jsonify({'success': False, 'error': 'image or image_path is required'}), 400
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
        
        # 自动保存原图和JSON文件
        save_result = None
        if image_path:
            try:
                # 准备保存参数
                save_params = {
                    'image_path': image_path,
                    'detections': results,
                    'params': {
                        'threshold': threshold,
                        'polygon_refinement': polygon_refinement,
                        'mask_iou_threshold': mask_iou_threshold,
                        'polygon_simplify_epsilon': poly_simplify_eps,
                        'polygon_collinear_epsilon': poly_collinear_eps
                    },
                    'save_subdir': 'auto_save'  # 自动保存到 results/auto_save 目录
                }
                
                # 调用保存逻辑 - 使用file_utils模块
                save_result = file_utils.save_segmentation_result(
                    image_path=save_params.get('image_path'),
                    detections=save_params.get('detections') or [],
                    params=save_params.get('params') or {},
                    save_subdir=save_params.get('save_subdir') or ''
                )
            except Exception as e:
                print(f"自动保存失败: {str(e)}")
                save_result = {'success': False, 'error': str(e)}

        return jsonify({
            'success': True,
            'image': image_base64,
            'detections': results,
            'auto_save_result': save_result
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

@app.route('/api/crawl', methods=['POST'])
def crawl_api():
    """图片爬虫：
    接收 JSON { query, engine, limit, out_dir(optional) }
    - engine: bing | baidu (google 暂不支持)
    - limit: 下载数量，默认 30
    - out_dir: 保存目录，若为相对路径，将相对于 RESULTS_FOLDER
    返回下载明细。
    """
    try:
        data = request.get_json() or {}
        query = (data.get('query') or '').strip()
        engine = (data.get('engine') or 'bing').lower()
        try:
            limit = int(data.get('limit', 30))
        except Exception:
            limit = 30
        out_dir = data.get('out_dir') or ''

        if not query:
            return jsonify({'success': False, 'error': 'query required'}), 400

        # 解析保存目录：相对路径则放到 RESULTS_FOLDER 下
        if out_dir:
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(RESULTS_FOLDER, out_dir)
        else:
            # 默认 results/auto_crawl/<query>
            safe_query = crawler.sanitize_filename(query) or 'images'
            out_dir = os.path.join(RESULTS_FOLDER, 'auto_crawl', safe_query)

        os.makedirs(out_dir, exist_ok=True)

        # 新任务开始前清除停止标记
        crawler.clear_stop()
        result = crawler.crawl_images(engine=engine, query=query, limit=limit, out_dir=out_dir)
        # 为前端提供相对 results 路径，便于展示
        try:
            rel_to_results = os.path.relpath(out_dir, RESULTS_FOLDER)
        except Exception:
            rel_to_results = out_dir
        result['out_dir'] = out_dir
        result['out_dir_rel'] = rel_to_results
        # 为每个保存的文件添加相对路径与可访问URL
        for item in result.get('saved', []) or []:
            try:
                rel = os.path.relpath(item.get('path', ''), RESULTS_FOLDER)
            except Exception:
                rel = item.get('path', '')
            item['rel'] = rel
            try:
                item['url'] = url_for('serve_results', filename=rel, _external=False)
            except Exception:
                item['url'] = ''
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/crawl/stop', methods=['POST'])
def crawl_stop_api():
    """请求停止当前爬虫任务（若在进行中）"""
    try:
        crawler.request_stop()
        return jsonify({'success': True, 'message': '已请求停止'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def _sse_event(event: str, data: dict) -> str:
    import json
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

@app.route('/api/crawl/stream', methods=['GET'])
def crawl_stream_api():
    """SSE: 边爬取边推送保存的图片
    查询参数: query, engine, limit, out_dir(可选, 相对results/或绝对)
    事件:
      - saved: {rel, url, src_url, index}
      - done:  {success, saved_count, found, out_dir_rel, out_dir, stopped}
    """
    @stream_with_context
    def generate():
        try:
            engine = (request.args.get('engine') or 'bing').lower()
            query = (request.args.get('query') or '').strip()
            try:
                limit = int(request.args.get('limit') or 30)
            except Exception:
                limit = 30
            out_dir = (request.args.get('out_dir') or '').strip()

            if not query:
                yield _sse_event('done', {'success': False, 'error': 'query required'})
                return

            # 输出目录规范化
            if out_dir:
                if not os.path.isabs(out_dir):
                    out_dir = os.path.join(RESULTS_FOLDER, out_dir)
            else:
                safe_query = crawler.sanitize_filename(query) or 'images'
                out_dir = os.path.join(RESULTS_FOLDER, 'auto_crawl', safe_query)
            os.makedirs(out_dir, exist_ok=True)

            # 清除停止标记
            crawler.clear_stop()

            # 获取链接列表
            if engine not in crawler.SUPPORTED_ENGINES:
                yield _sse_event('done', {'success': False, 'error': f'Unsupported engine: {engine}'})
                return
            fetcher = crawler.SUPPORTED_ENGINES[engine]
            urls = fetcher(query, limit)
            if not urls:
                yield _sse_event('done', {'success': False, 'error': 'No images found or search failed'})
                return

            saved_count = 0
            base = crawler.sanitize_filename(query) or 'images'
            for idx, u in enumerate(urls, start=1):
                if crawler.STOP_EVENT.is_set():
                    break
                ext = os.path.splitext(u.split('?')[0])[1].lower()
                if ext not in ['.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp']:
                    ext = '.jpg'
                fname = f"{base}_{idx:04d}{ext}"
                path = os.path.join(out_dir, fname)
                ok = crawler._download(u, path)
                if ok:
                    saved_count += 1
                    try:
                        rel = os.path.relpath(path, RESULTS_FOLDER)
                    except Exception:
                        rel = path
                    try:
                        file_url = url_for('serve_results', filename=rel, _external=False)
                    except Exception:
                        file_url = ''
                    yield _sse_event('saved', {
                        'rel': rel,
                        'url': file_url,
                        'src_url': u,
                        'index': idx,
                    })
                # 小延时 + 停止检查
                for _ in range(3):
                    if crawler.STOP_EVENT.is_set():
                        break
                    time.sleep(random.uniform(0.05, 0.1))

            try:
                out_dir_rel = os.path.relpath(out_dir, RESULTS_FOLDER)
            except Exception:
                out_dir_rel = out_dir
            yield _sse_event('done', {
                'success': True,
                'engine': engine,
                'query': query,
                'requested': limit,
                'found': len(urls),
                'saved_count': saved_count,
                'out_dir': out_dir,
                'out_dir_rel': out_dir_rel,
                'stopped': crawler.STOP_EVENT.is_set(),
            })
        except Exception as e:
            yield _sse_event('done', {'success': False, 'error': str(e)})

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',  # 兼容某些反向代理禁用缓冲
    }
    return Response(generate(), headers=headers)

@app.route('/api/config/folders', methods=['GET', 'POST'])
def config_folders_api():
    """获取或设置文件夹配置
    
    GET: 获取当前文件夹配置
    POST: 设置新的文件夹配置
    """
    global UPLOAD_FOLDER, RESULTS_FOLDER
    
    if request.method == 'GET':
        # 返回当前配置
        folders = file_utils.get_current_folders()
        return jsonify({
            'success': True,
            'folders': folders
        })
    else:
        # 设置新配置
        data = request.get_json()
        upload_folder = data.get('upload_folder')
        results_folder = data.get('results_folder')
        
        # 验证路径安全性
        if upload_folder and not os.path.isabs(upload_folder):
            upload_folder = os.path.join(BASE_DIR, upload_folder)
        
        if results_folder and not os.path.isabs(results_folder):
            results_folder = os.path.join(BASE_DIR, results_folder)
            
        # 配置文件夹
        result = file_utils.configure_folders(upload_folder, results_folder)
        
        # 更新全局变量
        UPLOAD_FOLDER = result['upload_folder']
        RESULTS_FOLDER = result['results_folder']
        
        return jsonify({
            'success': True,
            'message': '文件夹配置已更新',
            'folders': result
        })

@app.route('/api/upload', methods=['POST'])
def upload_api():
    """上传图片（多文件）并保存至 uploads/，返回可访问的URL与相对路径。
    - 当提供 form 字段 relative_paths 时，按相对路径保存（保留子目录结构），允许覆盖。
    - 否则，沿用旧逻辑：随机后缀防重名，平铺保存。
    """
    if 'files' not in request.files:
        return jsonify({'success': False, 'error': 'no files field'}), 400
    
    files = request.files.getlist('files')
    rel_list = request.form.getlist('relative_paths') or []
    use_rel = len(rel_list) == len(files) and len(files) > 0
    saved = []
    
    if use_rel:
        # 使用相对路径保存文件
        for idx, f in enumerate(files):
            if not f or not f.filename:
                continue
            rel_path = file_utils.sanitize_path(rel_list[idx] or f.filename)
            result = file_utils.save_uploaded_files([f], rel_path)
            if result.get('success') and result.get('files'):
                for file_info in result['files']:
                    file_info['url'] = url_for('serve_uploads', filename=file_info['path'], _external=False)
                saved.extend(result.get('files', []))
    else:
        # 使用随机文件名保存文件
        result = file_utils.save_uploaded_files(files)
        if result.get('success') and result.get('files'):
            for file_info in result['files']:
                file_info['url'] = url_for('serve_uploads', filename=file_info['path'], _external=False)
            saved = result.get('files', [])
    
    return jsonify({'success': True, 'files': saved})

@app.route('/uploads/<path:filename>')
def serve_uploads(filename):
    """静态访问上传的文件"""
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/results/<path:filename>')
def serve_results(filename):
    """静态访问 results 下的文件"""
    return send_from_directory(RESULTS_FOLDER, filename)

@app.route('/api/crawl/delete', methods=['POST'])
def crawl_delete_api():
    """删除 results 下的某个图片文件
    请求JSON: { path: 相对results/的路径 }
    """
    try:
        data = request.get_json() or {}
        rel_path = (data.get('path') or '').strip()
        if not rel_path:
            return jsonify({'success': False, 'error': 'path required'}), 400
        # 安全拼接路径，防止越权
        norm_rel = os.path.normpath(rel_path)
        abs_path = os.path.abspath(os.path.join(RESULTS_FOLDER, norm_rel))
        if not abs_path.startswith(os.path.abspath(RESULTS_FOLDER)):
            return jsonify({'success': False, 'error': 'invalid path'}), 400
        if not os.path.isfile(abs_path):
            return jsonify({'success': False, 'error': 'file not found'}), 404
        os.remove(abs_path)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/save_result', methods=['POST'])
def save_result_api():
    """保存分割结果：
    - 允许指定 save_subdir（相对于 results/）
    - 按 uploads 下的相对子目录结构保存（便于回溯）
    - 复制原图到结果目录，并生成与原图同名的 Labelme 风格 JSON
    """
    try:
        data = request.get_json()
        image_path = data.get('image_path')
        if not image_path:
            return jsonify({'success': False, 'error': 'image_path required'}), 400
        
        # 使用file_utils模块保存结果
        result = file_utils.save_segmentation_result(
            image_path=image_path,
            detections=data.get('detections') or [],
            params=data.get('params') or {},
            save_subdir=data.get('save_subdir') or ''
        )
        
        if result['success']:
            return jsonify(result)
        else:
            return jsonify(result), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# ======================================================================================================================
# =================================================== 数据增强 API ===================================================
# ======================================================================================================================

def _augment_sse_event(event: str, data: dict) -> str:
    return f"event: {event}\n" + "data: " + json.dumps(data, ensure_ascii=False) + "\n\n"

def _load_image_from_any_path(path: str) -> Optional[Image.Image]:
    """安全地从任意路径加载图像，确保路径是文件且存在"""
    abs_path = os.path.abspath(path)
    if not os.path.isfile(abs_path):
        return None
    try:
        return Image.open(abs_path).convert("RGBA")
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

def augment_and_save(sample_dir: str, bg_dir: str, class_name: str, out_dir: str):
    """数据增强核心逻辑：分割、提取、随机组合并保存"""
    try:
        # 1. 准备工作：查找图片、创建输出目录
        yield _augment_sse_event('progress', {'message': '正在准备文件...', 'progress': 5})
        sample_files = file_utils.find_image_files(sample_dir)
        bg_files = file_utils.find_image_files(bg_dir)
        os.makedirs(out_dir, exist_ok=True)

        if not sample_files or not bg_files:
            raise ValueError("样本或背景文件夹中没有找到图片")

        # 2. 从样本中提取目标
        yield _augment_sse_event('progress', {'message': f'正在从{len(sample_files)}个样本中提取 {class_name}...', 'progress': 15})
        extracted_objects = []
        for i, sample_path in enumerate(sample_files):
            if AUGMENT_STOP_EVENT.is_set(): return
            
            with open(sample_path, 'rb') as f:
                img_data = f.read()
            
            _, detections = grounded_segmentation(img_data, [class_name], threshold=0.4, mask_iou_threshold=0.7)
            
            for det in detections:
                if det.label.strip('.') == class_name and det.mask is not None:
                    img = Image.open(io.BytesIO(img_data)).convert("RGBA")
                    mask = Image.fromarray(det.mask.astype(np.uint8) * 255).convert('L')
                    
                    # 裁切 bbox 区域，应用蒙版
                    obj_img = Image.new("RGBA", img.size)
                    obj_img.paste(img, mask=mask)
                    bbox = (det.box.xmin, det.box.ymin, det.box.xmax, det.box.ymax)
                    cropped_obj = obj_img.crop(bbox)
                    extracted_objects.append(cropped_obj)
            
            progress = 15 + int(35 * (i + 1) / len(sample_files))
            yield _augment_sse_event('progress', {'message': f'已处理 {i+1}/{len(sample_files)} 个样本', 'progress': progress})

        if not extracted_objects:
            raise ValueError(f"在样本中未能分割出任何 '{class_name}' 物体")

        # 3. 生成增强图片
        yield _augment_sse_event('progress', {'message': f'共提取到 {len(extracted_objects)} 个物体，开始生成增强图片...', 'progress': 50})
        num_to_generate = min(len(extracted_objects) * 5, 200) # 最多生成200张
        
        for i in range(num_to_generate):
            if AUGMENT_STOP_EVENT.is_set(): return

            bg_path = random.choice(bg_files)
            background = _load_image_from_any_path(bg_path)
            if not background:
                continue
            
            # 随机选择一个或多个物体进行粘贴
            num_objects = random.randint(1, min(len(extracted_objects), 3))
            selected_objects = random.sample(extracted_objects, num_objects)
            
            new_detections = []
            current_bg = background.copy()

            for obj_img in selected_objects:
                # 随机缩放
                scale = random.uniform(0.5, 1.2)
                new_size = (int(obj_img.width * scale), int(obj_img.height * scale))
                if new_size[0] == 0 or new_size[1] == 0: continue
                resized_obj = obj_img.resize(new_size, Image.LANCZOS)

                # 随机位置
                max_x = current_bg.width - resized_obj.width
                max_y = current_bg.height - resized_obj.height
                if max_x <= 0 or max_y <= 0: continue
                paste_x = random.randint(0, max_x)
                paste_y = random.randint(0, max_y)

                # 粘贴
                current_bg.paste(resized_obj, (paste_x, paste_y), resized_obj)

                # 保存标注信息
                mask_np = np.array(resized_obj.split()[-1]) > 0
                new_detections.append({
                    'label': class_name,
                    'polygon': file_utils.mask_to_polygon(mask_np, epsilon=0.8, offset=(paste_x, paste_y)),
                    'shape_type': 'polygon',
                })

            # 保存图片和标注
            img_filename = f"augmented_{i+1:04d}.jpg"
            json_filename = f"augmented_{i+1:04d}.json"
            img_path = os.path.join(out_dir, img_filename)
            json_path = os.path.join(out_dir, json_filename)

            current_bg.convert("RGB").save(img_path, 'JPEG', quality=95)
            
            labelme_data = {
                'version': '5.0.1',
                'flags': {},
                'shapes': new_detections,
                'imagePath': img_filename,
                'imageData': None,
                'imageHeight': current_bg.height,
                'imageWidth': current_bg.width,
            }
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)

            progress = 50 + int(50 * (i + 1) / num_to_generate)
            yield _augment_sse_event('progress', {'message': f'已生成 {i+1}/{num_to_generate} 张增强图片', 'progress': progress})

        yield _augment_sse_event('done', {'success': True, 'message': f'数据增强完成，共生成 {num_to_generate} 张图片。'})

    except Exception as e:
        yield _augment_sse_event('done', {'success': False, 'error': str(e)})

@app.route('/api/augment/stream', methods=['GET'])
def augment_stream_api():
    """SSE: 数据增强流式API"""
    @stream_with_context
    def generate():
        try:
            sample_dir = request.args.get('sample_dir')
            bg_dir = request.args.get('bg_dir')
            class_name = request.args.get('class_name')
            out_dir = request.args.get('out_dir')

            if not all([sample_dir, bg_dir, class_name]):
                yield _augment_sse_event('done', {'success': False, 'error': '缺少必要参数'})
                return

            if not out_dir:
                safe_name = file_utils.sanitize_filename(class_name) or 'augmented'
                out_dir = os.path.join(RESULTS_FOLDER, 'augmented_data', safe_name)
            
            AUGMENT_STOP_EVENT.clear()
            yield from augment_and_save(sample_dir, bg_dir, class_name, out_dir)

        except Exception as e:
            yield _augment_sse_event('done', {'success': False, 'error': str(e)})

    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'X-Accel-Buffering': 'no',
    }
    return Response(stream_with_context(generate()), headers=headers)

@app.route('/api/augment/stop', methods=['POST'])
def augment_stop_api():
    """请求停止数据增强任务"""
    try:
        AUGMENT_STOP_EVENT.set()
        return jsonify({'success': True, 'message': '已请求停止数据增强任务'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False, threaded=True)
    except KeyboardInterrupt:
        print('\n接收到键盘中断，正在关闭...')
        # 确保资源被释放
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
