# UniPixel快速开始指南

## 🚀 5分钟快速启动

### 步骤1：安装依赖

```bash
# 运行自动安装脚本
bash setup_unipixel.sh
```

或手动安装：

```bash
cd UniPixel-main
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
cd ..
```

### 步骤2：配置引擎

```bash
# 设置使用UniPixel引擎
export INFERENCE_ENGINE=unipixel
```

### 步骤3：运行测试

```bash
# 运行集成测试
python test_unipixel_integration.py
```

### 步骤4：启动应用

```bash
# 启动Flask应用
python app.py
```

访问：http://localhost:5000

## 📝 API使用示例

### JavaScript示例

```javascript
// 使用UniPixel引擎
const response = await fetch('/api/segment', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
        image: imageBase64,
        labels: ['cat', 'dog', 'person'],
        threshold: 0.3,
        engine: 'unipixel'  // 指定使用UniPixel
    })
});

const result = await response.json();
console.log(result.detections);
```

### Python示例

```python
import requests
import base64

# 读取图片
with open('test.jpg', 'rb') as f:
    image_b64 = base64.b64encode(f.read()).decode()

# 调用API
response = requests.post('http://localhost:5000/api/segment', json={
    'image': f'data:image/jpeg;base64,{image_b64}',
    'labels': ['cat', 'dog'],
    'threshold': 0.3,
    'engine': 'unipixel'
})

result = response.json()
print(f"检测到 {len(result['detections'])} 个对象")
```

## 🔄 引擎对比

### 使用UniPixel（新）

```python
engine = 'unipixel'
# 优势：
# - 端到端模型
# - 更强的语言理解
# - 支持复杂prompt
```

### 使用Grounding DINO + SAM（原有）

```python
engine = 'grounding_dino_sam'
# 特点：
# - 两阶段流程
# - 成熟稳定
# - 资源占用较小
```

## 💡 最佳实践

1. **首次使用**：模型会自动从HuggingFace下载（约2-3GB）
2. **GPU推荐**：使用GPU能显著提升速度
3. **内存需求**：3B模型需要约6-8GB VRAM
4. **Prompt优化**：使用更具体的描述可以提升效果

## 🎯 高级Prompt示例

```python
# 基础
labels = ['cat']

# 带位置描述
labels = ['the cat on the left']

# 带属性描述  
labels = ['the largest dog', 'the red ball']

# 复杂描述
labels = ['the person wearing glasses sitting on the chair']
```

## 📊 性能提示

| 场景 | 推荐引擎 |
|------|---------|
| 简单物体检测 | Grounding DINO + SAM |
| 复杂场景理解 | UniPixel |
| 批量处理 | 两者都可 |
| 资源受限 | Grounding DINO + SAM |
| 需要高精度 | UniPixel |

## 🐛 常见问题

**Q: 如何切换引擎？**

A: 在API请求中添加 `"engine": "unipixel"` 或 `"engine": "grounding_dino_sam"`

**Q: 模型下载在哪里？**

A: 默认在 `~/.cache/huggingface/hub/`

**Q: 如何使用本地模型？**

A: 设置 `export UNIPIXEL_MODEL_PATH=/path/to/local/model`

**Q: CPU模式能用吗？**

A: 可以，但会比较慢。推荐使用GPU。

## 📚 更多资源

- 详细文档：`UNIPIXEL_INTEGRATION.md`
- 测试脚本：`test_unipixel_integration.py`
- UniPixel官方：https://github.com/PolyU-ChenLab/UniPixel
