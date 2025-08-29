#!/usr/bin/env python3
"""
SAM任意物体分割应用测试脚本
"""

import requests
import base64
import json
import time
from PIL import Image
import numpy as np

def create_test_image():
    """创建一个简单的测试图像"""
    # 创建一个简单的测试图像
    img_array = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # 保存到内存
    import io
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    img_data = buffer.getvalue()
    
    return base64.b64encode(img_data).decode('utf-8')

def test_health_endpoint():
    """测试健康检查端点"""
    print("🔍 测试健康检查端点...")
    try:
        response = requests.get('http://localhost:5000/api/health', timeout=5)
        if response.status_code == 200:
            print("✅ 健康检查通过")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 健康检查失败: {e}")
        return False

def test_segment_endpoint():
    """测试分割端点"""
    print("🔍 测试分割端点...")
    
    # 创建测试数据
    test_image = create_test_image()
    test_data = {
        "image": f"data:image/png;base64,{test_image}",
        "labels": ["test object"],
        "threshold": 0.3,
        "polygon_refinement": True
    }
    
    try:
        response = requests.post(
            'http://localhost:5000/api/segment',
            json=test_data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print("✅ 分割端点测试通过")
                print(f"   检测到 {len(result.get('detections', []))} 个对象")
                return True
            else:
                print(f"❌ 分割失败: {result.get('error', '未知错误')}")
                return False
        else:
            print(f"❌ 分割端点测试失败: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 分割端点测试失败: {e}")
        return False

def test_frontend():
    """测试前端页面"""
    print("🔍 测试前端页面...")
    try:
        response = requests.get('http://localhost:5000/', timeout=5)
        if response.status_code == 200:
            print("✅ 前端页面可访问")
            return True
        else:
            print(f"❌ 前端页面访问失败: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ 前端页面访问失败: {e}")
        return False

def main():
    """主测试函数"""
    print("🧪 SAM任意物体分割应用测试")
    print("=" * 50)
    
    # 等待应用启动
    print("⏳ 等待应用启动...")
    time.sleep(3)
    
    # 测试健康检查
    if not test_health_endpoint():
        print("❌ 应用可能未正常启动")
        return
    
    # 测试前端页面
    if not test_frontend():
        print("❌ 前端页面无法访问")
        return
    
    # 测试分割功能
    if not test_segment_endpoint():
        print("❌ 分割功能测试失败")
        return
    
    print("\n" + "=" * 50)
    print("🎉 所有测试通过！应用运行正常")
    print("📱 请在浏览器中访问 http://localhost:5000 使用应用")

if __name__ == "__main__":
    main()
