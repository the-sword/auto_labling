"""
UniPixel集成测试脚本
验证UniPixel推理引擎是否正常工作
"""

import os
import sys
from PIL import Image
import numpy as np

print("=" * 60)
print("UniPixel Integration Test")
print("=" * 60)

# 测试1: 检查依赖
print("\n[Test 1] 检查依赖导入...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"✗ PyTorch导入失败: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except ImportError as e:
    print(f"✗ Transformers导入失败: {e}")
    sys.exit(1)

# 测试2: 检查UniPixel模块
print("\n[Test 2] 检查UniPixel模块...")
try:
    from unipixel_inference import UniPixelInference, UNIPIXEL_AVAILABLE
    if not UNIPIXEL_AVAILABLE:
        print("✗ UniPixel依赖未安装")
        sys.exit(1)
    print("✓ UniPixel模块导入成功")
except ImportError as e:
    print(f"✗ UniPixel模块导入失败: {e}")
    sys.exit(1)

# 测试3: 检查配置
print("\n[Test 3] 检查配置...")
try:
    import inference_config
    engine = inference_config.get_inference_engine()
    print(f"✓ 当前推理引擎: {engine}")
    print(f"  UniPixel模型路径: {inference_config.UNIPIXEL_MODEL_PATH}")
except ImportError as e:
    print(f"✗ 配置导入失败: {e}")
    sys.exit(1)

# 测试4: 检查统一推理接口
print("\n[Test 4] 检查统一推理接口...")
try:
    from unified_inference import UnifiedInferenceEngine
    print("✓ 统一推理接口导入成功")
except ImportError as e:
    print(f"✗ 统一推理接口导入失败: {e}")
    sys.exit(1)

# 测试5: 尝试加载模型（可选，需要模型文件）
print("\n[Test 5] 模型加载测试...")
load_model = input("是否尝试加载UniPixel模型？(y/n，首次会下载模型): ").strip().lower()

if load_model == 'y':
    try:
        print("正在加载模型，请稍候...")
        from unipixel_inference import get_unipixel_model
        
        # 使用较小的3B模型进行测试
        model = get_unipixel_model("PolyU-ChenLab/UniPixel-3B")
        print("✓ UniPixel模型加载成功")
        
        # 测试推理
        test_inference = input("是否进行推理测试？需要提供测试图片 (y/n): ").strip().lower()
        if test_inference == 'y':
            image_path = input("请输入测试图片路径: ").strip()
            if os.path.exists(image_path):
                print(f"正在处理图片: {image_path}")
                image = Image.open(image_path).convert("RGB")
                
                # 简单的分割测试
                labels = input("请输入标签（逗号分隔，如: cat,dog）: ").strip().split(',')
                labels = [l.strip() for l in labels if l.strip()]
                
                print(f"执行分割，标签: {labels}")
                results = model.segment(image, labels, threshold=0.3)
                
                print(f"✓ 推理完成！检测到 {len(results)} 个对象")
                for i, res in enumerate(results):
                    print(f"  [{i+1}] {res.label}: 置信度={res.score:.2f}, "
                          f"bbox=({res.box.xmin},{res.box.ymin})-({res.box.xmax},{res.box.ymax})")
                    if res.mask is not None:
                        print(f"      mask shape={res.mask.shape}")
            else:
                print(f"✗ 图片不存在: {image_path}")
        
    except Exception as e:
        print(f"✗ 模型加载或推理失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("跳过模型加载测试")

# 总结
print("\n" + "=" * 60)
print("测试总结")
print("=" * 60)
print("✓ 所有基础测试通过")
print("\n下一步：")
print("1. 启动Flask应用: python app.py")
print("2. 访问 http://localhost:5000")
print("3. 在API请求中添加 'engine': 'unipixel' 使用新引擎")
print("\n详细文档请查看: UNIPIXEL_INTEGRATION.md")
print("=" * 60)
