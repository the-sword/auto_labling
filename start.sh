#!/bin/bash

# SAM任意物体分割应用快速启动脚本

echo "🚀 SAM任意物体分割应用启动脚本"
echo "=================================="

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 未找到Python3，请先安装Python 3.8+"
    exit 1
fi

# 检查pip是否安装
if ! command -v pip &> /dev/null; then
    echo "❌ 错误: 未找到pip，请先安装pip"
    exit 1
fi

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "📦 创建虚拟环境..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "🔧 激活虚拟环境..."
source venv/bin/activate

# 安装依赖
echo "📥 安装依赖包..."
pip install -r requirements.txt

# 启动应用
echo "🎯 启动应用..."
python run.py
