# SAM任意物体分割 - Grounded Segment Anything

基于Grounding DINO和Segment Anything Model (SAM)的文本引导图像分割Web应用。

## 功能特点

- 🎯 **文本引导分割**: 通过自然语言描述来分割图像中的目标对象
- 🖼️ **拖拽上传**: 支持拖拽和点击上传图片
- 🏷️ **多标签支持**: 可以同时检测和分割多个目标对象
- ⚙️ **参数调节**: 可调节检测阈值和多边形优化选项
- 🎨 **可视化结果**: 实时显示分割结果和边界框
- 📱 **响应式设计**: 支持桌面和移动设备

## 技术栈

### 后端
- **Flask**: Web框架
- **Transformers**: Hugging Face模型库
- **Grounding DINO**: 零样本目标检测
- **SAM**: Segment Anything Model
- **OpenCV**: 图像处理
- **PyTorch**: 深度学习框架

### 前端
- **HTML5**: 页面结构
- **CSS3**: 样式设计
- **JavaScript**: 交互逻辑
- **Bootstrap 5**: UI组件库
- **Font Awesome**: 图标库

## 安装和运行

### 1. 环境要求

- Python 3.8+
- CUDA支持（可选，用于GPU加速）

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行应用

```bash
python app.py
```

应用将在 `http://localhost:5000` 启动。

## 使用说明

### 1. 上传图片
- 点击上传区域选择图片
- 或直接拖拽图片到上传区域

### 2. 添加标签
- 在标签输入框中输入目标对象描述
- 点击"+"按钮或按回车键添加标签
- 也可以点击示例标签快速添加

### 3. 调整参数
- **检测阈值**: 控制检测的敏感度（0.1-0.9）
- **多边形优化**: 启用/禁用分割结果的边界优化

### 4. 开始分割
- 点击"开始分割"按钮
- 等待处理完成（首次运行需要下载模型）

### 5. 查看结果
- 原图和分割结果并排显示
- 检测结果列表显示详细信息
- 可以清除结果重新开始

## 模型说明

### Grounding DINO
- 模型: `IDEA-Research/grounding-dino-tiny`
- 功能: 零样本目标检测
- 输入: 图像 + 文本描述
- 输出: 边界框坐标

### SAM (Segment Anything Model)
- 模型: `facebook/sam-vit-base`
- 功能: 图像分割
- 输入: 图像 + 边界框
- 输出: 分割掩码

## API接口

### POST /api/segment

执行图像分割

**请求参数:**
```json
{
    "image": "base64编码的图像数据",
    "labels": ["标签1", "标签2"],
    "threshold": 0.3,
    "polygon_refinement": true
}
```

**响应:**
```json
{
    "success": true,
    "image": "base64编码的原图",
    "detections": [
        {
            "score": 0.95,
            "label": "cat",
            "box": {
                "xmin": 100,
                "ymin": 50,
                "xmax": 300,
                "ymax": 250
            },
            "mask": "base64编码的分割掩码",
            "polygon": [[x1, y1], [x2, y2], ...]
        }
    ]
}
```

### GET /api/health

健康检查接口

## 项目结构

```
auto_labling/
├── app.py                 # Flask应用主文件
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── templates/
│   └── index.html        # 主页面模板
└── static/
    ├── css/
    │   └── style.css     # 样式文件
    └── js/
        └── app.js        # JavaScript逻辑
```

## 注意事项

1. **首次运行**: 首次运行时会自动下载模型文件，需要网络连接
2. **内存要求**: SAM模型较大，建议至少8GB内存
3. **GPU加速**: 如果有CUDA支持的GPU，会自动使用GPU加速
4. **图片格式**: 支持JPG、PNG等常见图片格式
5. **处理时间**: 根据图片大小和标签数量，处理时间在几秒到几十秒不等

## 故障排除

### 常见问题

1. **模型下载失败**
   - 检查网络连接
   - 确保有足够的磁盘空间

2. **内存不足**
   - 关闭其他应用程序
   - 使用更小的图片

3. **CUDA错误**
   - 检查CUDA版本兼容性
   - 尝试使用CPU模式

### 日志查看

应用运行时会输出详细的日志信息，包括：
- 模型加载状态
- 处理进度
- 错误信息

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 致谢

- [IDEA\-Research/grounding\-dino\-tiny · Hugging Face](https://huggingface.co/IDEA-Research/grounding-dino-tiny)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [facebook/sam\-vit\-base · Hugging Face](https://huggingface.co/facebook/sam-vit-base)
