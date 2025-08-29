"""
文件处理工具模块
提供图像和分割结果的保存、加载和处理功能
"""
import os
import io
import base64
import json
import shutil
import random
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional, Union, Tuple

# 常量定义
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def simplify_polygon(points: List[List[int]], epsilon: float = 2.0, collinear_eps: float = 1.0) -> List[List[int]]:
    """简化多边形点集：
    - 使用 Ramer–Douglas–Peucker（approxPolyDP）按像素误差 epsilon 简化
    - 进一步删除共线点（相邻三点构成的三角形面积 < collinear_eps）
    """
    import cv2
    
    if not points or len(points) < 3:
        return points
    
    # 转换为numpy数组
    points_np = np.array(points, dtype=np.int32)
    
    # 使用RDP算法简化
    epsilon = max(0.1, float(epsilon))  # 防止epsilon太小导致过度简化
    approx = cv2.approxPolyDP(points_np.reshape(-1, 1, 2), epsilon, True)
    simplified = approx.reshape(-1, 2).tolist()
    
    # 如果简化后点数过少，直接返回
    if len(simplified) < 4:
        return simplified
    
    # 移除共线点
    if collinear_eps > 0:
        filtered = []
        n = len(simplified)
        for i in range(n):
            p1 = simplified[i]
            p2 = simplified[(i + 1) % n]
            p3 = simplified[(i + 2) % n]
            
            # 计算三角形面积
            area = abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
            
            # 如果面积小于阈值，则p2是共线点，可以移除
            if area >= collinear_eps:
                if not filtered or filtered[-1] != p1:
                    filtered.append(p1)
            else:
                if not filtered or filtered[-1] != p1:
                    filtered.append(p1)
        
        # 如果过滤后的点数足够，则使用过滤后的结果
        if len(filtered) >= 3:
            return filtered
    
    return simplified

def mask_to_polygon(mask: np.ndarray, epsilon: float = 2.0, collinear_eps: float = 1.0) -> List[List[int]]:
    """将mask转换为简化后的多边形坐标"""
    import cv2
    
    if mask is None:
        return []
    
    # 确保mask是二值图像
    binary = (mask > 0).astype(np.uint8) * 255
    
    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果没有轮廓，返回空列表
    if not contours:
        return []
    
    # 找到最大轮廓
    max_contour = max(contours, key=cv2.contourArea)
    
    # 将轮廓转换为点列表
    points = max_contour.reshape(-1, 2).tolist()
    
    # 简化多边形
    return simplify_polygon(points, epsilon, collinear_eps)

def sanitize_path(path: str) -> str:
    """清理路径，防止路径遍历攻击"""
    # 替换所有反斜杠为正斜杠
    path = path.replace('\\', '/')
    
    # 分割路径并过滤掉危险部分
    parts = path.split('/')
    safe_parts = []
    for part in parts:
        # 过滤掉空部分、当前目录和父目录
        if part and part != '.' and part != '..':
            # 只保留字母数字和安全字符
            safe = ''.join(c for c in part if c.isalnum() or c in (' ', '.', '-', '_')).strip()
            if safe:
                safe_parts.append(safe)
    
    return '/'.join(safe_parts)

def sanitize_subdir(subdir: str) -> str:
    """清理子目录名，确保安全"""
    if not subdir:
        return ''
    
    # 清理子目录名
    safe_sub = ''.join(c for c in subdir if c.isalnum() or c in (' ', '.', '-', '_', '/')).strip().replace(' ', '_')
    safe_sub = safe_sub.replace('\\', '/').lstrip('/')
    
    return safe_sub

def ensure_dir_exists(directory: str) -> None:
    """确保目录存在，如不存在则创建"""
    os.makedirs(directory, exist_ok=True)

def is_path_safe(path: str, base_dir: str) -> bool:
    """检查路径是否在安全目录内"""
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(base_dir)
    return abs_path.startswith(abs_base) and os.path.exists(abs_path)

def get_image_data(image_path: str) -> Tuple[Image.Image, str, int, int]:
    """获取图像数据、格式和尺寸"""
    with Image.open(image_path) as im:
        width, height = im.size
        im_format = 'JPEG' if im.format == 'JPEG' else 'PNG'
        # 创建副本以避免文件关闭后无法访问
        image_copy = im.copy()
    
    return image_copy, im_format, width, height

def image_to_base64(image: Image.Image, format: str = 'PNG') -> str:
    """将PIL图像转换为base64编码"""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def save_segmentation_result(
    image_path: str,
    detections: List[Dict[str, Any]],
    params: Dict[str, Any] = None,
    save_subdir: str = ''
) -> Dict[str, Any]:
    """
    保存分割结果：
    - 允许指定 save_subdir（相对于 results/）
    - 按 uploads 下的相对子目录结构保存（便于回溯）
    - 复制原图到结果目录，并生成与原图同名的 Labelme 风格 JSON
    
    Args:
        image_path: 图像路径（相对于uploads目录）
        detections: 检测结果列表
        params: 分割参数
        save_subdir: 保存子目录（相对于results目录）
        
    Returns:
        Dict: 包含保存结果的字典
    """
    try:
        # 参数预处理
        params = params or {}
        
        # 验证图像路径在 uploads 内
        abs_image = os.path.abspath(os.path.join(UPLOAD_FOLDER, image_path)) if not os.path.isabs(image_path) else os.path.abspath(image_path)
        if not is_path_safe(abs_image, os.path.abspath(UPLOAD_FOLDER)):
            return {'success': False, 'error': 'invalid image_path'}
        
        # 目的根目录
        dest_root = RESULTS_FOLDER
        if save_subdir:
            safe_sub = sanitize_subdir(save_subdir)
            dest_root = os.path.join(RESULTS_FOLDER, safe_sub)
        
        # 相对 uploads 的子路径
        rel_subdir = os.path.dirname(image_path).replace('\\', '/')
        dest_dir = os.path.join(dest_root, rel_subdir)
        ensure_dir_exists(dest_dir)
        
        # 复制原图
        img_basename = os.path.basename(abs_image)
        dest_image_path = os.path.join(dest_dir, img_basename)
        shutil.copyfile(abs_image, dest_image_path)
        
        # 读取图像尺寸与数据
        image, im_format, width, height = get_image_data(abs_image)
        image_b64 = image_to_base64(image, im_format)
        
        # 解析 detections -> shapes（优先 polygon；如无则从 mask 还原多边形）
        try:
            poly_eps = float(params.get('polygon_simplify_epsilon', 2.0))
        except Exception:
            poly_eps = 2.0
        try:
            col_eps = float(params.get('polygon_collinear_epsilon', 1.0))
        except Exception:
            col_eps = 1.0
        
        shapes = []
        for det in detections:
            label = str(det.get('label', 'object'))
            polygon = det.get('polygon')
            if (not polygon or len(polygon) < 3) and det.get('mask'):
                try:
                    mask_b64 = det['mask']
                    mask_bytes = base64.b64decode(mask_b64)
                    m = Image.open(io.BytesIO(mask_bytes)).convert('L')
                    mask_np = np.array(m) > 0
                    poly = mask_to_polygon(mask_np.astype(np.uint8), epsilon=poly_eps, collinear_eps=col_eps)
                    polygon = poly
                except Exception:
                    polygon = []
            if not polygon or len(polygon) < 3:
                # 跳过无效多边形
                continue
            shapes.append({
                'mask': None,
                'label': label,
                'points': [[float(p[0]), float(p[1])] for p in polygon],
                'group_id': None,
                'description': '',
                'shape_type': 'polygon',
                'flags': {}
            })
        
        labelme_payload = {
            'version': '5.3.1',
            'flags': {},
            'shapes': shapes,
            'imagePath': img_basename,
            'imageData': image_b64,
            'imageHeight': int(height),
            'imageWidth': int(width)
        }
        
        stem = os.path.splitext(img_basename)[0]
        json_path = os.path.join(dest_dir, f'{stem}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_payload, f, ensure_ascii=False)
        
        result_rel = os.path.relpath(json_path, RESULTS_FOLDER)
        return {'success': True, 'result_path': result_rel}
    except Exception as e:
        return {'success': False, 'error': str(e)}

def save_uploaded_files(files, rel_path=''):
    """
    保存上传的文件
    
    Args:
        files: 上传的文件列表
        rel_path: 相对保存路径
        
    Returns:
        Dict: 包含保存结果的字典
    """
    def sanitize_filename(filename):
        return ''.join(c for c in filename if c.isalnum() or c in (' ', '.', '-', '_')).strip().replace(' ', '_')
    
    saved = []
    for idx, f in enumerate(files):
        if not f or not f.filename:
            continue
        
        name = f.filename
        if rel_path:
            # 新逻辑：按照指定的相对路径保存
            rel_path = rel_path.replace('\\', '/')
            if rel_path.endswith('/') or rel_path == '':
                base = sanitize_filename(name) or 'image.png'
                rel_path = (rel_path.rstrip('/') + '/' if rel_path else '') + base
            save_path = os.path.join(UPLOAD_FOLDER, rel_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            f.save(save_path)
            saved.append({
                'name': name,
                'rel_path': rel_path,
                'path': rel_path,
                'url': f'/uploads/{rel_path}',
                'size': os.path.getsize(save_path)
            })
        else:
            # 旧逻辑：随机后缀，平铺
            safe_name = sanitize_filename(name)
            stem, ext = os.path.splitext(safe_name)
            suffix = str(random.randint(10000, 99999))
            fname = f"{stem}_{suffix}{ext or '.png'}"
            save_path = os.path.join(UPLOAD_FOLDER, fname)
            f.save(save_path)
            rel_path = fname
            saved.append({
                'name': name,
                'saved_name': fname,
                'path': rel_path,
                'url': f'/uploads/{rel_path}',
                'size': os.path.getsize(save_path)
            })
    return {'success': True, 'files': saved}
