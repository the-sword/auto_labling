# UniPixel集成指南

本文档说明如何在自动标注系统中使用UniPixel推理引擎。

## 📋 概述

已将[UniPixel](https://github.com/PolyU-ChenLab/UniPixel)集成到自动标注系统中，作为Grounding DINO + SAM的替代方案。

**UniPixel优势**：
- 🎯 端到端统一模型，无需分离的检测和分割步骤
- 🚀 基于Qwen2.5-VL的强大多模态理解能力
- 🎨 支持复杂的自然语言描述
- 📊 在多个benchmark上表现优秀

## 🛠️ 安装依赖

### 1. 安装UniPixel依赖

```bash
# 进入UniPixel目录
cd UniPixel-main

# 安装PyTorch (根据你的CUDA版本调整)
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装Flash Attention (推荐，提升性能)
pip install flash_attn==2.8.2 --no-build-isolation

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 下载模型

UniPixel需要预训练模型。你有两个选择：

**选项1：使用HuggingFace自动下载（推荐）**
- 模型会在首次运行时自动从HuggingFace下载
- 默认使用 `PolyU-ChenLab/UniPixel-3B`

**选项2：手动下载**
```bash
# 下载UniPixel-3B模型
git lfs install
git clone https://huggingface.co/PolyU-ChenLab/UniPixel-3B

# 或下载UniPixel-7B模型（更强但更慢）
git clone https://huggingface.co/PolyU-ChenLab/UniPixel-7B
```

## ⚙️ 配置

### 环境变量配置

在启动应用前设置环境变量：

```bash
# 选择推理引擎：'unipixel' 或 'grounding_dino_sam'
export INFERENCE_ENGINE=unipixel

# 指定UniPixel模型路径（可选，默认使用HuggingFace ID）
export UNIPIXEL_MODEL_PATH=PolyU-ChenLab/UniPixel-3B
# 或使用本地路径
# export UNIPIXEL_MODEL_PATH=/path/to/UniPixel-3B
```

### 代码中配置

在 `inference_config.py` 中修改：

```python
# 推理引擎选择
INFERENCE_ENGINE = 'unipixel'  # 或 'grounding_dino_sam'

# UniPixel模型路径
UNIPIXEL_MODEL_PATH = 'PolyU-ChenLab/UniPixel-3B'
```

## 🚀 使用方法

### 1. 启动应用

```bash
# 使用UniPixel引擎启动
export INFERENCE_ENGINE=unipixel
python app.py
```

### 2. API调用

API端点 `/api/segment` 现在支持 `engine` 参数：

```javascript
// 使用UniPixel
fetch('/api/segment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        image: imageBase64,
        labels: ['cat', 'dog'],
        threshold: 0.3,
        engine: 'unipixel'  // 指定使用UniPixel
    })
})

// 使用原有的Grounding DINO + SAM
fetch('/api/segment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        image: imageBase64,
        labels: ['cat', 'dog'],
        threshold: 0.3,
        engine: 'grounding_dino_sam'  // 使用原引擎
    })
})

// 不指定engine则使用配置的默认引擎
```

### 3. Python代码调用

```python
from unified_inference import get_unified_engine
from PIL import Image

# 获取统一推理引擎
engine = get_unified_engine()

# 单张图片分割
image = Image.open('example.jpg')
labels = ['cat', 'dog']
results = engine.segment(image, labels, threshold=0.3)

# 批量处理
images = [Image.open(f'image{i}.jpg') for i in range(5)]
results_batch = engine.segment_batch(images, labels, threshold=0.3)
```

## 🔄 引擎切换

### 运行时切换

```python
import inference_config

# 切换到UniPixel
inference_config.set_inference_engine('unipixel')

# 切换到Grounding DINO + SAM
inference_config.set_inference_engine('grounding_dino_sam')

# 重新加载引擎
from unified_inference import get_unified_engine
engine = get_unified_engine()
engine.reload()
```

## 📊 性能对比

| 特性 | Grounding DINO + SAM | UniPixel |
|------|---------------------|----------|
| **推理速度** | 中等 | 中等-快 |
| **内存占用** | 中等 | 高（3B模型约6-8GB VRAM） |
| **检测精度** | 高 | 很高 |
| **分割质量** | 高 | 很高 |
| **复杂prompt支持** | 基础 | 强大 |
| **零样本能力** | 好 | 很好 |

## 🐛 故障排除

### 问题1：导入错误

```
ImportError: cannot import name 'build_model' from 'unipixel.model.builder'
```

**解决方案**：
- 检查UniPixel-main目录是否存在
- 确保已安装所有依赖

### 问题2：CUDA内存不足

```
RuntimeError: CUDA out of memory
```

**解决方案**：
- 使用CPU模式：设置 `device='cpu'`
- 使用更小的模型：切换到UniPixel-3B
- 减少batch size

### 问题3：模型下载失败

**解决方案**：
- 检查网络连接
- 手动下载模型并设置本地路径
- 配置HuggingFace镜像

## 📝 示例Prompt

UniPixel支持更复杂的自然语言描述：

```python
# 简单描述
"Please segment the cat."

# 具体描述
"Please segment the tallest giraffe."

# 位置描述
"Where is the nearest sheep? Please provide the segmentation mask."

# 带推理的描述
"Why is the boy crying? Please provide the segmentation mask and explain why."

# 多对象
"Please segment all instances of cats and dogs."
```

## 🔗 相关资源

- [UniPixel GitHub](https://github.com/PolyU-ChenLab/UniPixel)
- [UniPixel Paper](https://arxiv.org/abs/2509.18094)
- [UniPixel HuggingFace](https://huggingface.co/collections/PolyU-ChenLab/unipixel-68cf7137013455e5b15962e8)
- [UniPixel Demo](https://huggingface.co/spaces/PolyU-ChenLab/UniPixel)

## 📞 支持

遇到问题请：
1. 检查上述故障排除部分
2. 查看UniPixel官方文档
3. 提交Issue到项目仓库
