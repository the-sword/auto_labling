#!/bin/bash

# UniPixel集成安装脚本
# 用于快速设置UniPixel推理引擎

set -e

echo "================================================"
echo "  UniPixel Integration Setup"
echo "================================================"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 检查Python版本
echo -e "${YELLOW}[1/5] 检查Python版本...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "当前Python版本: $PYTHON_VERSION"

# 检查CUDA
echo -e "${YELLOW}[2/5] 检查CUDA...${NC}"
if command -v nvidia-smi &> /dev/null; then
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    echo "检测到CUDA版本: $CUDA_VERSION"
    USE_GPU=true
else
    echo -e "${RED}未检测到CUDA，将使用CPU模式${NC}"
    USE_GPU=false
fi

# 进入UniPixel目录
echo -e "${YELLOW}[3/5] 进入UniPixel目录...${NC}"
cd UniPixel-main

# 安装PyTorch
echo -e "${YELLOW}[4/5] 安装PyTorch...${NC}"
if [ "$USE_GPU" = true ]; then
    echo "安装GPU版本PyTorch..."
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
    
    # 安装Flash Attention (可选但推荐)
    echo "安装Flash Attention..."
    pip install flash_attn==2.8.2 --no-build-isolation || {
        echo -e "${YELLOW}Flash Attention安装失败，继续使用标准attention${NC}"
    }
else
    echo "安装CPU版本PyTorch..."
    pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo -e "${YELLOW}[5/5] 安装UniPixel依赖...${NC}"
pip install -r requirements.txt

# 返回主目录
cd ..

# 配置环境变量
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}✓ UniPixel依赖安装完成！${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo "下一步操作："
echo ""
echo "1. 设置环境变量启用UniPixel："
echo "   export INFERENCE_ENGINE=unipixel"
echo ""
echo "2. (可选) 指定模型路径："
echo "   export UNIPIXEL_MODEL_PATH=PolyU-ChenLab/UniPixel-3B"
echo ""
echo "3. 启动应用："
echo "   python app.py"
echo ""
echo "查看完整文档: UNIPIXEL_INTEGRATION.md"
echo ""
