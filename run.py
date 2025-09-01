#!/usr/bin/env python3
"""
SAM任意物体分割应用启动脚本
"""

import sys
import os
import subprocess
import importlib.util

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    print(f"✅ Python版本检查通过: {sys.version}")
    return True

def check_dependencies():
    """检查依赖包"""
    required_packages = [
        'torch', 'transformers', 'flask',
        'PIL', 'numpy', 'requests'
    ]

    missing_packages = []

    for package in required_packages:
        if package == 'PIL':
            spec = importlib.util.find_spec('PIL')
        else:
            spec = importlib.util.find_spec(package)

        if spec is None:
            missing_packages.append(package)

    if missing_packages:
        print("❌ 缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装依赖:")
        print("pip install -r requirements.txt")
        return False

    print("✅ 依赖包检查通过")
    return True

def check_cuda():
    """检查CUDA可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU模式")
            return True
    except ImportError:
        print("⚠️  无法检查CUDA状态")
        return True

def create_directories():
    """创建必要的目录"""
    directories = ['templates', 'static/css', 'static/js']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("✅ 目录结构检查完成")

def main():
    """主函数"""
    print("🚀 SAM任意物体分割应用启动器")
    print("=" * 50)

    # 检查Python版本
    if not check_python_version():
        sys.exit(1)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 检查CUDA
    check_cuda()

    # 创建目录
    create_directories()

    print("\n" + "=" * 50)
    print("🎯 启动应用...")
    print("📱 应用将在 http://localhost:5000 启动")
    print("🛑 按 Ctrl+C 停止应用")
    print("=" * 50)

    try:
        # 启动Flask应用
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
