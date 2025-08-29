// 全局变量
let currentImage = null;
let currentLabels = [];
let detectionResults = [];

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const imageContainer = document.getElementById('imageContainer');
const labelInput = document.getElementById('labelInput');
const addLabelBtn = document.getElementById('addLabelBtn');
const labelsList = document.getElementById('labelsList');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const polygonRefinement = document.getElementById('polygonRefinement');
const segmentBtn = document.getElementById('segmentBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const resultCanvas = document.getElementById('resultCanvas');
const detectionsList = document.getElementById('detectionsList');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));

// 预定义标签（可点击快速添加）
const PREDEFINED_LABELS = [
    'obstacle',
    'stool',
    'fence',
    'aldult',
    'pet',
    'leaf',
    'charging station',
    'manhole cover',
    'water',
    'flagstone',
    'Flat spray can',
    'pipeline',
    'mud',
    'child',
    'hedgehog',
    'fruilt',
    'green plants',
    'grass',
    'road',
    'background'
];

// 颜色生成函数
function generateRandomColor() {
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// 预填充目标标签列表
function prefillTargetLabels() {
    currentLabels = [...PREDEFINED_LABELS];
    updateLabelsDisplay();
    updateSegmentButton();
}

// 渲染示例标签（基于预定义列表）
function renderExampleTags() {
    const container = document.querySelector('.example-tags');
    if (!container) return;
    // 仅当容器为空时再渲染，避免覆盖服务端已渲染的内容
    if (container.children.length === 0) {
        container.innerHTML = PREDEFINED_LABELS
            .map(l => `<span class="badge bg-light text-dark me-1 mb-1 example-tag">${l}</span>`)
            .join('');
    }
}

// 初始化事件监听器
function initializeEventListeners() {
    // 图片上传
    uploadArea.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    imageInput.addEventListener('change', handleImageSelect);

    // 标签管理
    addLabelBtn.addEventListener('click', addLabel);
    labelInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            addLabel();
        }
    });

    // 示例标签点击（事件代理，兼容服务端/前端渲染）
    const exampleTagsContainer = document.querySelector('.example-tags');
    if (exampleTagsContainer) {
        exampleTagsContainer.addEventListener('click', (e) => {
            const tag = e.target.closest('.example-tag');
            if (!tag) return;
            const label = tag.textContent.trim();
            if (!currentLabels.includes(label)) {
                addLabelToArray(label);
                updateLabelsDisplay();
            }
        });
    }

    // 参数设置
    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = e.target.value;
    });

    // 操作按钮
    segmentBtn.addEventListener('click', performSegmentation);
    clearBtn.addEventListener('click', clearResults);
}

// 拖拽处理
function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        const file = files[0];
        if (file.type.startsWith('image/')) {
            loadImage(file);
        }
    }
}

// 图片选择处理
function handleImageSelect(e) {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
}

// 加载图片
function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        currentImage = e.target.result;
        displayImage(currentImage);
        updateSegmentButton();
    };
    reader.readAsDataURL(file);
}

// 显示图片
function displayImage(imageSrc) {
    imageContainer.innerHTML = `<img src="${imageSrc}" alt="上传的图片" style="max-width: 100%; max-height: 100%; object-fit: contain;">`;
}

// 标签管理
function addLabel() {
    const label = labelInput.value.trim();
    if (label && !currentLabels.includes(label)) {
        addLabelToArray(label);
        updateLabelsDisplay();
        labelInput.value = '';
    }
}

function addLabelToArray(label) {
    currentLabels.push(label);
    updateSegmentButton();
}

function removeLabel(index) {
    currentLabels.splice(index, 1);
    updateLabelsDisplay();
    updateSegmentButton();
}

function updateLabelsDisplay() {
    labelsList.innerHTML = '';
    currentLabels.forEach((label, index) => {
        const labelItem = document.createElement('div');
        labelItem.className = 'label-item fade-in';
        labelItem.innerHTML = `
            <span class="badge">${label}</span>
            <button class="remove-label" onclick="removeLabel(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        labelsList.appendChild(labelItem);
    });
}

// 更新分割按钮状态
function updateSegmentButton() {
    segmentBtn.disabled = !currentImage || currentLabels.length === 0;
}

// 执行分割
async function performSegmentation() {
    if (!currentImage || currentLabels.length === 0) {
        showError('请先上传图片并添加标签');
        return;
    }

    showLoading(true);

    try {
        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                image: currentImage,
                labels: currentLabels,
                threshold: parseFloat(thresholdSlider.value),
                polygon_refinement: polygonRefinement.checked
            })
        });

        const data = await response.json();

        if (data.success) {
            detectionResults = data.detections;
            displayResults(data.image, data.detections);
        } else {
            showError(data.error || '分割失败');
        }
    } catch (error) {
        showError('网络错误: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// 显示结果
function displayResults(imageSrc, detections) {
    // 显示原图
    originalImage.src = `data:image/png;base64,${imageSrc}`;

    // 绘制分割结果
    drawSegmentationResult(imageSrc, detections);

    // 显示检测结果列表
    displayDetectionsList(detections);

    // 显示结果区域
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// 绘制分割结果
async function drawSegmentationResult(imageSrc, detections) {
    const canvas = resultCanvas;
    const ctx = canvas.getContext('2d');

    const loadImage = (src) => new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = src;
    });

    try {
        const mainImage = await loadImage(`data:image/png;base64,${imageSrc}`);
        canvas.width = mainImage.width;
        canvas.height = mainImage.height;
        ctx.drawImage(mainImage, 0, 0);

        for (const detection of detections) {
            //print
            console.log(detection);

            if (detection.mask) {
                const maskImage = await loadImage(`data:image/png;base64,${detection.mask}`);

                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = mainImage.width;
                tempCanvas.height = mainImage.height;
                tempCtx.drawImage(maskImage, 0, 0);

                const maskData = tempCtx.getImageData(0, 0, mainImage.width, mainImage.height);
                const colorHex = generateRandomColor();
                const colorRgb = hexToRgb(colorHex);

                for (let i = 0; i < maskData.data.length; i += 4) {
                    const isMask = maskData.data[i] > 0; // red channel > 0
                    if (isMask) {
                        maskData.data[i] = colorRgb.r;
                        maskData.data[i + 1] = colorRgb.g;
                        maskData.data[i + 2] = colorRgb.b;
                        maskData.data[i + 3] = 128; // semi-transparent for mask
                    } else {
                        // Make background fully transparent
                        maskData.data[i + 3] = 0;
                    }
                }
                tempCtx.putImageData(maskData, 0, 0);

                ctx.drawImage(tempCanvas, 0, 0);

                // Draw bounding box and label
                const box = detection.box;
                ctx.strokeStyle = colorHex;
                ctx.lineWidth = 2;
                ctx.strokeRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);

                ctx.fillStyle = colorHex;
                ctx.font = '14px Arial';
                ctx.fillText(`${detection.label}: ${detection.score.toFixed(2)}`, box.xmin, box.ymin - 5);
            }
        }
    } catch (error) {
        console.error('Failed to draw segmentation result:', error);
        showError('Failed to render segmentation masks.');
    }
}

// 显示检测结果列表
function displayDetectionsList(detections) {
    detectionsList.innerHTML = '';

    detections.forEach((detection, index) => {
        const detectionItem = document.createElement('div');
        detectionItem.className = 'detection-item fade-in';
        detectionItem.innerHTML = `
            <div class="detection-info">
                <div class="detection-label">${detection.label}</div>
                <div class="detection-score">置信度: ${(detection.score * 100).toFixed(1)}%</div>
            </div>
            <div class="detection-box">
                [${detection.box.xmin}, ${detection.box.ymin}, ${detection.box.xmax}, ${detection.box.ymax}]
            </div>
        `;
        detectionsList.appendChild(detectionItem);
    });
}

// 清除结果
function clearResults() {
    currentImage = null;
    currentLabels = [];
    detectionResults = [];

    imageContainer.innerHTML = `
        <div class="placeholder-text">
            <i class="fas fa-image fa-4x mb-3"></i>
            <h4>请上传图片开始分割</h4>
            <p class="text-muted">支持JPG、PNG等常见图片格式</p>
        </div>
    `;

    labelsList.innerHTML = '';
    resultsSection.style.display = 'none';
    updateSegmentButton();
}

// 工具函数
function hexToRgb(hex) {
    const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorModal.show();
}

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', () => {
    // 先渲染示例标签，再绑定事件
    renderExampleTags();
    // 预填充目标标签列表
    prefillTargetLabels();
    initializeEventListeners();
    updateSegmentButton();
});
