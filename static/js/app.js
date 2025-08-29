// 全局变量
let currentImage = null;
let currentLabels = [];
let detectionResults = [];
let selectedDetectionIndex = null; // 当前选中的分割结果索引
let isDraggingVertex = false;      // 是否在拖拽多边形顶点
let draggingVertexIndex = -1;      // 被拖拽的顶点索引
let lastResultImageBase64 = null;  // 最近一次结果图像（用于重绘）
let hoveredDetectionIndex = null;  // 悬浮的分割结果索引（用于联动高亮）

// 功能开关：检测列表与画布的悬浮联动
const ENABLE_HOVER_LINK = false;
// 功能开关：启用本地存储持久化
const ENABLE_LOCAL_STORAGE = true;
// localStorage键前缀
const STORAGE_KEY_PREFIX = 'auto_labeling_';

// 多图队列
let imageQueue = []; // [{ name, file, dataURL (lazy), relPath, serverPath, serverUrl }]
let currentImageIndex = -1;
let perImageResults = new Map(); // key: index, value: { detections, resultImageBase64 }

// 上传图片文件到后台，更新队列条目的 serverPath/serverUrl
async function uploadFiles(files) {
    if (!files || files.length === 0) return;
    const formData = new FormData();
    const rels = [];
    for (const f of files) {
        formData.append('files', f);
        const rp = (f.webkitRelativePath && f.webkitRelativePath.length > 0) ? f.webkitRelativePath : f.name;
        rels.push(rp);
    }
    // 并行传递相对路径，后端按顺序对应
    for (const rp of rels) formData.append('relative_paths', rp);
    showLoading(true);
    try {
        const resp = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });
        const data = await resp.json();
        if (!data.success) {
            showError(data.error || '上传失败');
            return;
        }
        // 以相对路径优先匹配，写回队列项
        const returned = data.files || [];
        for (const item of returned) {
            const retRel = item.path || item.rel_path || '';
            let q = imageQueue.find(qi => qi.relPath === retRel);
            if (!q) {
                // 回退到按名称匹配（可能有同名风险）
                q = imageQueue.find(qi => qi.name === item.name && !qi.serverPath);
            }
            if (q) {
                q.serverPath = item.path; // 相对 uploads/
                q.serverUrl = item.url;   // /uploads/<name>
            }
        }
        // 可能影响保存按钮状态
        updateSegmentButton();
    } catch (err) {
        showError('上传失败: ' + (err?.message || err));
    } finally {
        showLoading(false);
    }
}

// 保存当前图片的分割结果到后台
async function saveCurrentResults() {
    const item = imageQueue[currentImageIndex] || null;
    if (!item || !item.serverPath || !Array.isArray(detectionResults) || detectionResults.length === 0) {
        showError('无可保存的结果或图片未上传到服务器');
        return;
    }
    showLoading(true);
    try {
        const resp = await fetch('/api/save_result', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_path: item.serverPath,
                detections: detectionResults,
                params: {
                    threshold: parseFloat(thresholdSlider.value),
                    polygon_refinement: polygonRefinement.checked,
                    mask_iou_threshold: maskIouSlider ? parseFloat(maskIouSlider.value) : 0.5,
                    polygon_simplify_epsilon: polyEpsSlider ? parseFloat(polyEpsSlider.value) : 2.0,
                    polygon_collinear_epsilon: polyCollinearSlider ? parseFloat(polyCollinearSlider.value) : 1.0
                },
                save_subdir: (typeof saveSubdirInput !== 'undefined' && saveSubdirInput) ? (saveSubdirInput.value || '') : ''
            })
        });
        const data = await resp.json();
        if (!data.success) {
            showError(data.error || '保存失败');
            return;
        }
        // 简单提示
        alert('保存成功');
    } catch (err) {
        showError('保存失败: ' + (err?.message || err));
    } finally {
        showLoading(false);
        // 保存后再次刷新保存按钮（保持一致）
        updateSegmentButton();
    }
}

// 在画布上绘制带圆角背景的文本，提升可读性
function drawTextWithBackground(ctx, text, x, y, options = {}) {
    const {
        font = 'bold 14px Arial',
        paddingX = 6,
        paddingY = 3,
        radius = 6,
        bgColor = 'rgba(0,0,0,0.6)',
        textColor = '#fff',
        align = 'left',
        baseline = 'top',
        shadow = true
    } = options;

    ctx.save();
    ctx.font = font;
    ctx.textAlign = align;
    ctx.textBaseline = baseline;
    const metrics = ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = Math.max(parseInt(font.match(/(\d+)px/)[1] || 14, 10), 10);

    // 计算背景框位置（基于左上角）
    let bx = x, by = y;
    if (align === 'center') bx = x - (textWidth / 2) - paddingX;
    else if (align === 'right') bx = x - textWidth - paddingX * 2;
    const bw = textWidth + paddingX * 2;
    const bh = textHeight + paddingY * 2;

    // 背景圆角矩形
    ctx.beginPath();
    const r = Math.min(radius, bh / 2);
    ctx.moveTo(bx + r, by);
    ctx.arcTo(bx + bw, by, bx + bw, by + bh, r);
    ctx.arcTo(bx + bw, by + bh, bx, by + bh, r);
    ctx.arcTo(bx, by + bh, bx, by, r);
    ctx.arcTo(bx, by, bx + bw, by, r);
    ctx.closePath();
    ctx.fillStyle = bgColor;
    ctx.fill();

    // 文本
    if (shadow) {
        ctx.shadowColor = 'rgba(0,0,0,0.6)';
        ctx.shadowBlur = 2;
        ctx.shadowOffsetX = 0;
        ctx.shadowOffsetY = 1;
    }
    ctx.fillStyle = textColor;
    ctx.fillText(text, bx + paddingX, by + paddingY);
    ctx.restore();
}

// 同步结果列表的悬浮高亮
function updateDetectionListHover() {
    if (!detectionsList) return;
    const items = detectionsList.querySelectorAll('.detection-item');
    if (!ENABLE_HOVER_LINK) {
        // 关闭联动：移除所有 hovered 样式
        items.forEach(el => el.classList.remove('hovered'));
        return;
    }
    items.forEach(el => {
        const idx = parseInt(el.getAttribute('data-index'));
        if (Number.isInteger(idx)) {
            el.classList.toggle('hovered', idx === hoveredDetectionIndex);
        }
    });
}

function setHoveredDetection(index) {
    if (!ENABLE_HOVER_LINK) return; // 关闭联动
    const next = (Number.isInteger(index) && index >= 0 && index < detectionResults.length) ? index : null;
    if (next === hoveredDetectionIndex) return;
    hoveredDetectionIndex = next;
    updateDetectionListHover();
    redraw();
}

// 视图变换（缩放/平移）
let viewScale = 1;
let viewOffsetX = 0;
let viewOffsetY = 0;
let isPanning = false;
let lastPanClient = { x: 0, y: 0 };
let spacePressed = false;

// 离屏底图（减少频繁重绘导致的闪烁）
let baseRenderCanvas = document.createElement('canvas');
let baseRenderReady = false;

// DOM元素
const uploadArea = document.getElementById('uploadArea');
const imageInput = document.getElementById('imageInput');
const labelInput = document.getElementById('labelInput');
const addLabelBtn = document.getElementById('addLabelBtn');
const labelsList = document.getElementById('labelsList');
const thresholdSlider = document.getElementById('thresholdSlider');
const thresholdValue = document.getElementById('thresholdValue');
const maskIouSlider = document.getElementById('maskIouSlider');
const maskIouValue = document.getElementById('maskIouValue');
const polyEpsSlider = document.getElementById('polyEpsSlider');
const polyEpsValue = document.getElementById('polyEpsValue');
const polyCollinearSlider = document.getElementById('polyCollinearSlider');
const polyCollinearValue = document.getElementById('polyCollinearValue');
const polygonRefinement = document.getElementById('polygonRefinement');
const segmentBtn = document.getElementById('segmentBtn');
const clearBtn = document.getElementById('clearBtn');
const resultsSection = document.getElementById('resultsSection');
const originalImage = document.getElementById('originalImage');
const originalPlaceholder = document.getElementById('originalPlaceholder');
const resultCanvas = document.getElementById('resultCanvas');
const detectionsList = document.getElementById('detectionsList');
const loadingOverlay = document.getElementById('loadingOverlay');
const errorModal = new bootstrap.Modal(document.getElementById('errorModal'));
const annotateToggleBtn = document.getElementById('annotateToggleBtn');
const helpBtn = document.getElementById('helpBtn');
const helpModal = new bootstrap.Modal(document.getElementById('helpModal'));
const prevBtn = document.getElementById('prevBtn');
const nextBtn = document.getElementById('nextBtn');
const batchBtn = document.getElementById('batchBtn');
const queueInfo = document.getElementById('queueInfo');
const saveBtn = document.getElementById('saveBtn');
const saveSubdirInput = document.getElementById('saveSubdirInput');

// 手动标注状态
let isAnnotating = false;
let annotationPoints = [];

// 预定义标签（可点击快速添加）
const PREDEFINED_LABELS = [
    'obstacle',
    'dung',
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

// 规范化标签显示：去掉末尾句点（中英文）并去除首尾空白
function sanitizeLabel(label) {
    if (typeof label !== 'string') return label;
    return label.trim().replace(/[。．.]+$/u, '');
}

// 颜色生成函数
function generateRandomColor() {
    const colors = [
        '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
        '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
        '#F8C471', '#82E0AA', '#F1948A', '#85C1E9', '#D7BDE2'
    ];
    return colors[Math.floor(Math.random() * colors.length)];
}

// 批量分割：依次对队列中所有图片执行分割
async function runBatchSegmentation() {
    if (imageQueue.length <= 1) return;
    const startIndex = currentImageIndex >= 0 ? currentImageIndex : 0;
    showLoading(true);
    try {
        for (let i = 0; i < imageQueue.length; i++) {
            const idx = i; // 顺序处理
            navigateTo(idx);
            // 确保已加载dataURL
            await new Promise(resolve => {
                if (imageQueue[idx].dataURL) return resolve();
                const reader = new FileReader();
                reader.onload = (e) => {
                    imageQueue[idx].dataURL = e.target.result;
                    resolve();
                };
                reader.readAsDataURL(imageQueue[idx].file);
            });
            currentImage = imageQueue[idx].dataURL;
            await performSegmentation();
        }
    } catch (err) {
        showError('批量分割失败：' + (err?.message || err));
    } finally {
        showLoading(false);
        navigateTo(startIndex);
    }
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
    if (maskIouSlider) {
        maskIouSlider.addEventListener('input', (e) => {
            maskIouValue.textContent = e.target.value;
        });
        // 初始化显示
        maskIouValue.textContent = maskIouSlider.value;
    }
    if (polyEpsSlider) {
        polyEpsSlider.addEventListener('input', (e) => {
            if (polyEpsValue) polyEpsValue.textContent = e.target.value;
        });
        if (polyEpsValue) polyEpsValue.textContent = polyEpsSlider.value;
    }
    if (polyCollinearSlider) {
        polyCollinearSlider.addEventListener('input', (e) => {
            if (polyCollinearValue) polyCollinearValue.textContent = e.target.value;
        });
        if (polyCollinearValue) polyCollinearValue.textContent = polyCollinearSlider.value;
    }

    // 操作按钮
    segmentBtn.addEventListener('click', performSegmentation);
    clearBtn.addEventListener('click', clearResults);
    if (prevBtn) prevBtn.addEventListener('click', () => navigateTo(currentImageIndex - 1));
    if (nextBtn) nextBtn.addEventListener('click', () => navigateTo(currentImageIndex + 1));
    if (batchBtn) batchBtn.addEventListener('click', runBatchSegmentation);
    if (saveBtn) saveBtn.addEventListener('click', saveCurrentResults);

    // 手动标注开关
    if (annotateToggleBtn) {
        annotateToggleBtn.addEventListener('click', toggleAnnotationMode);
    }
    // 帮助按钮
    if (helpBtn) {
        helpBtn.addEventListener('click', () => helpModal.show());
    }

    // 结果列表悬浮联动（事件委托）- 受开关控制
    if (detectionsList && ENABLE_HOVER_LINK) {
        detectionsList.addEventListener('mouseover', (e) => {
            const item = e.target.closest('.detection-item');
            if (!item || !detectionsList.contains(item)) return;
            const idx = parseInt(item.getAttribute('data-index'));
            if (Number.isInteger(idx)) setHoveredDetection(idx);
        });
        detectionsList.addEventListener('mouseout', (e) => {
            const related = e.relatedTarget;
            if (related && detectionsList.contains(related)) return; // 仍在列表内
            setHoveredDetection(null);
        });
        // 当移出某个item但仍在列表区域时，尝试读取新的item
        detectionsList.addEventListener('mousemove', (e) => {
            const item = e.target.closest('.detection-item');
            if (!item) return;
            const idx = parseInt(item.getAttribute('data-index'));
            if (Number.isInteger(idx)) setHoveredDetection(idx);
        });
    }
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

    // 优先使用 entries 方式，以保留相对路径（目录拖拽）
    const items = e.dataTransfer.items;
    if (items && items.length) {
        collectFilesFromDataTransfer(items).then((files) => {
            const imageFiles = files.filter(f => (f.type && f.type.startsWith('image/')) || /\.(png|jpg|jpeg|bmp|gif|webp)$/i.test(f.name));
            if (imageFiles.length === 0) {
                showError('未检测到图片文件');
                return;
            }
            setImageQueue(imageFiles);
            uploadFiles(imageFiles).catch(err => console.error(err));
        }).catch(err => {
            console.warn('目录遍历失败，回退至 files 列表:', err);
            const files = Array.from(e.dataTransfer.files || []);
            const imageFiles = files.filter(f => f.type && f.type.startsWith('image/'));
            if (imageFiles.length === 0) {
                showError('未检测到图片文件');
                return;
            }
            setImageQueue(imageFiles);
            uploadFiles(imageFiles).catch(err2 => console.error(err2));
        });
        return;
    }

    // 回退：直接使用 files
    const files = Array.from(e.dataTransfer.files || []);
    const imageFiles = files.filter(f => f.type && f.type.startsWith('image/'));
    if (imageFiles.length === 0) {
        showError('未检测到图片文件');
        return;
    }
    setImageQueue(imageFiles);
    uploadFiles(imageFiles).catch(err => console.error(err));
}

// 图片选择处理
function handleImageSelect(e) {
    const files = Array.from(e.target.files || []);
    const imageFiles = files.filter(f => f.type && f.type.startsWith('image/'));
    if (imageFiles.length === 0) return;
    setImageQueue(imageFiles);
    uploadFiles(imageFiles).catch(err => console.error(err));
}

// 加载图片
function loadImageFromQueue(index) {
    const item = imageQueue[index];
    if (!item) return;
    if (item.dataURL) {
        currentImage = item.dataURL;
        displayImage(currentImage);
        restoreResultsForIndex(index);
        updateSegmentButton();
        return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
        item.dataURL = e.target.result;
        currentImage = item.dataURL;
        displayImage(currentImage);
        restoreResultsForIndex(index);
        updateSegmentButton();
    };
    reader.readAsDataURL(item.file);
}

function setImageQueue(files) {
    imageQueue = files.map(f => ({
        name: f.name,
        file: f,
        dataURL: null,
        relPath: (f.relativePath && f.relativePath.length > 0)
            ? f.relativePath
            : ((f.webkitRelativePath && f.webkitRelativePath.length > 0) ? f.webkitRelativePath : f.name)
    }));
    currentImageIndex = 0;
    perImageResults.clear();
    updateQueueInfo();
    updateNavButtons();
    loadImageFromQueue(currentImageIndex);
}

// 递归收集 DataTransferItemList 中的文件，保留相对路径
function collectFilesFromDataTransfer(items) {
    return new Promise((resolve, reject) => {
        const allFiles = [];
        let pending = 0;

        function readEntry(entry, pathPrefix) {
            if (!entry) return;
            if (entry.isFile) {
                pending++;
                entry.file(file => {
                    // 附加相对路径属性
                    const relPath = pathPrefix ? `${pathPrefix}/${file.name}` : file.name;
                    try {
                        Object.defineProperty(file, 'relativePath', { value: relPath, configurable: true });
                    } catch (_) {
                        file.relativePath = relPath;
                    }
                    allFiles.push(file);
                    pending--;
                    if (pending === 0) resolve(allFiles);
                }, err => {
                    pending--;
                    console.warn('读取文件失败:', err);
                    if (pending === 0) resolve(allFiles);
                });
            } else if (entry.isDirectory) {
                const reader = entry.createReader();
                function readEntries() {
                    pending++;
                    reader.readEntries(entries => {
                        pending--;
                        if (!entries || entries.length === 0) {
                            if (pending === 0) resolve(allFiles);
                            return;
                        }
                        for (const ent of entries) {
                            readEntry(ent, pathPrefix ? `${pathPrefix}/${entry.name}` : entry.name);
                        }
                        // 继续读取该目录剩余条目
                        readEntries();
                    }, err => {
                        console.warn('读取目录失败:', err);
                        if (pending === 0) resolve(allFiles);
                    });
                }
                readEntries();
            }
        }

        try {
            for (let i = 0; i < items.length; i++) {
                const item = items[i];
                const entry = item.webkitGetAsEntry ? item.webkitGetAsEntry() : null;
                if (entry) readEntry(entry, '');
            }
            if (pending === 0) resolve(allFiles);
        } catch (e) {
            reject(e);
        }
    });
}

function navigateTo(index) {
    if (index < 0 || index >= imageQueue.length) return;
    currentImageIndex = index;
    loadImageFromQueue(currentImageIndex);
    updateQueueInfo();
    updateNavButtons();
}

function updateQueueInfo() {
    if (!queueInfo) return;
    if (imageQueue.length <= 1) {
        queueInfo.style.display = 'none';
        queueInfo.textContent = '';
    } else {
        queueInfo.style.display = '';
        queueInfo.textContent = `已导入 ${imageQueue.length} 张，当前 ${currentImageIndex + 1}/${imageQueue.length}：${imageQueue[currentImageIndex]?.name || ''}`;
    }
}

function updateNavButtons() {
    const hasQueue = imageQueue.length > 0;
    if (prevBtn) prevBtn.disabled = !(hasQueue && currentImageIndex > 0);
    if (nextBtn) nextBtn.disabled = !(hasQueue && currentImageIndex < imageQueue.length - 1);
    if (batchBtn) batchBtn.disabled = imageQueue.length <= 1;
}

function restoreResultsForIndex(index) {
    // 切换图片时，恢复该图片的分割结果（若存在）
    detectionResults = [];
    selectedDetectionIndex = null;
    lastResultImageBase64 = null;
    baseRenderReady = false;
    resultsSection.style.display = 'none';
    detectionsList.innerHTML = '';
    
    // 先尝试从内存中获取
    const saved = perImageResults.get(index);
    if (saved) {
        detectionResults = JSON.parse(JSON.stringify(saved.detections || []));
        displayResults(saved.resultImageBase64 || null, detectionResults);
        return;
    }
    
    // 如果内存中没有，尝试从localStorage加载
    if (ENABLE_LOCAL_STORAGE) {
        const item = imageQueue[index];
        if (item && item.relPath) {
            const storageKey = getStorageKeyForImage(item.relPath);
            const storedData = localStorage.getItem(storageKey);
            if (storedData) {
                try {
                    const parsedData = JSON.parse(storedData);
                    detectionResults = parsedData.detections || [];
                    // 保存到内存中，避免重复从localStorage加载
                    perImageResults.set(index, {
                        detections: JSON.parse(JSON.stringify(detectionResults)),
                        resultImageBase64: parsedData.resultImageBase64
                    });
                    displayResults(parsedData.resultImageBase64 || null, detectionResults);
                } catch (err) {
                    console.error('Failed to parse stored results:', err);
                }
            }
        }
    }
}

// 显示图片
function displayImage(imageSrc) {
    // 直接显示到顶部的原图区域
    if (originalImage) {
        originalImage.src = imageSrc;
        originalImage.style.display = '';
    }
    if (originalPlaceholder) originalPlaceholder.style.display = 'none';
}

// 标签管理
function addLabel() {
    const label = sanitizeLabel(labelInput.value.trim());
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
    updateNavButtons();
    // 同时更新保存按钮状态：需要当前有服务器路径且有检测结果
    if (saveBtn) {
        const item = imageQueue[currentImageIndex] || null;
        const canSave = !!(item && item.serverPath && Array.isArray(detectionResults) && detectionResults.length > 0);
        saveBtn.disabled = !canSave;
    }
}

// 执行分割
async function performSegmentation() {
    if (!currentImage || currentLabels.length === 0) {
        showError('请先上传图片并添加标签');
        return;
    }

    showLoading(true);

    try {
        // 收集手动标注（没有mask且有polygon的项）
        const manualAnnotations = (detectionResults || [])
            .filter(d => Array.isArray(d.polygon) && d.polygon.length > 2 && !d.mask)
            .map(d => ({
                label: d.label,
                score: typeof d.score === 'number' ? d.score : 1.0,
                box: d.box,
                polygon: d.polygon,
                is_manual: true
            }));

        // 优先使用服务器路径，避免重复上传图像数据
        const queueItem = imageQueue[currentImageIndex] || {};
        const payload = {
            labels: currentLabels,
            threshold: parseFloat(thresholdSlider.value),
            polygon_refinement: polygonRefinement.checked,
            mask_iou_threshold: maskIouSlider ? parseFloat(maskIouSlider.value) : 0.5,
            polygon_simplify_epsilon: polyEpsSlider ? parseFloat(polyEpsSlider.value) : 2.0,
            polygon_collinear_epsilon: polyCollinearSlider ? parseFloat(polyCollinearSlider.value) : 1.0,
            manual_annotations: manualAnnotations
        };
        if (queueItem && queueItem.serverPath) {
            payload.image_path = queueItem.serverPath;
        } else {
            payload.image = currentImage;
        }

        const response = await fetch('/api/segment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (data.success) {
            detectionResults = data.detections;
            displayResults(data.image, data.detections);
            // 保存当前图片的结果
            if (currentImageIndex >= 0) {
                const resultData = {
                    detections: JSON.parse(JSON.stringify(detectionResults)),
                    resultImageBase64: data.image
                };
                
                // 保存到内存
                perImageResults.set(currentImageIndex, resultData);
                
                // 保存到localStorage
                if (ENABLE_LOCAL_STORAGE && queueItem && queueItem.relPath) {
                    saveResultsToLocalStorage(queueItem.relPath, resultData);
                }
                
                updateQueueInfo();
            }
            // 结果就绪后刷新按钮状态（启用保存按钮等）
            updateSegmentButton();
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
    originalImage.style.display = '';
    if (originalPlaceholder) originalPlaceholder.style.display = 'none';

    // 绘制分割结果
    lastResultImageBase64 = imageSrc;
    baseRenderReady = false;
    buildBaseLayer(imageSrc, detections).then(() => {
        drawFromBaseLayer();
    });

    // 显示检测结果列表
    displayDetectionsList(detections);

    // 显示结果区域
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // 启用画布交互
    enableCanvasInteractions();
}

// 构建离屏底图（图像 + mask + bbox + label + 非高亮多边形）
async function buildBaseLayer(imageSrc, detections) {
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
        // 调整主canvas尺寸
        canvas.width = mainImage.width;
        canvas.height = mainImage.height;
        // 准备离屏画布
        baseRenderCanvas.width = mainImage.width;
        baseRenderCanvas.height = mainImage.height;
        const bctx = baseRenderCanvas.getContext('2d');
        // 清空离屏
        bctx.setTransform(1, 0, 0, 1, 0, 0);
        bctx.clearRect(0, 0, baseRenderCanvas.width, baseRenderCanvas.height);
        bctx.drawImage(mainImage, 0, 0);

        for (let idx = 0; idx < detections.length; idx++) {
            const detection = detections[idx];
            //print
            console.log(detection);

            // 为每个目标确定颜色
            const colorHex = generateRandomColor();
            const colorRgb = hexToRgb(colorHex);

            if (detection.mask) {
                // 有服务端mask：按像素着色
                const maskImage = await loadImage(`data:image/png;base64,${detection.mask}`);

                const tempCanvas = document.createElement('canvas');
                const tempCtx = tempCanvas.getContext('2d');
                tempCanvas.width = mainImage.width;
                tempCanvas.height = mainImage.height;
                tempCtx.drawImage(maskImage, 0, 0);

                const maskData = tempCtx.getImageData(0, 0, mainImage.width, mainImage.height);

                for (let i = 0; i < maskData.data.length; i += 4) {
                    const isMask = maskData.data[i] > 0; // red channel > 0
                    if (isMask) {
                        maskData.data[i] = colorRgb.r;
                        maskData.data[i + 1] = colorRgb.g;
                        maskData.data[i + 2] = colorRgb.b;
                        maskData.data[i + 3] = 128; // semi-transparent for mask
                    } else {
                        maskData.data[i + 3] = 0; // fully transparent
                    }
                }
                tempCtx.putImageData(maskData, 0, 0);
                bctx.drawImage(tempCanvas, 0, 0);
            } else if (Array.isArray(detection.polygon) && detection.polygon.length > 2) {
                // 无mask但有多边形（手动标注）：填充半透明区域
                bctx.save();
                bctx.beginPath();
                detection.polygon.forEach((pt, i) => {
                    if (i === 0) bctx.moveTo(pt[0], pt[1]);
                    else bctx.lineTo(pt[0], pt[1]);
                });
                bctx.closePath();
                bctx.fillStyle = `rgba(${colorRgb.r}, ${colorRgb.g}, ${colorRgb.b}, 0.35)`;
                bctx.fill();
                bctx.restore();
            }

            // 无论是否有mask，都绘制bbox与标签
            if (detection.box) {
                const box = detection.box;
                bctx.strokeStyle = colorHex;
                bctx.lineWidth = 2;
                bctx.strokeRect(box.xmin, box.ymin, box.xmax - box.xmin, box.ymax - box.ymin);

                const scoreText = typeof detection.score === 'number' ? detection.score.toFixed(2) : '1.00';
                const labelText = `${sanitizeLabel(detection.label)}: ${scoreText}`;
                drawTextWithBackground(bctx, labelText, box.xmin, Math.max(2, box.ymin - 18), {
                    font: 'bold 14px Arial',
                    bgColor: 'rgba(0,0,0,0.55)',
                    textColor: '#FFFFFF',
                    paddingX: 6,
                    paddingY: 3,
                    radius: 6,
                    align: 'left',
                    baseline: 'top',
                    shadow: true
                });
            }

            // 画多边形轮廓（如果有）
            if (Array.isArray(detection.polygon) && detection.polygon.length > 2) {
                bctx.save();
                bctx.beginPath();
                bctx.lineWidth = 2;
                bctx.strokeStyle = '#00FF88';
                detection.polygon.forEach((pt, i) => {
                    if (i === 0) bctx.moveTo(pt[0], pt[1]);
                    else bctx.lineTo(pt[0], pt[1]);
                });
                bctx.closePath();
                bctx.stroke();
                bctx.restore();
            }
        }
        baseRenderReady = true;
    } catch (error) {
        console.error('Failed to draw segmentation result:', error);
        showError('Failed to render segmentation masks.');
    }
}

// 将离屏底图绘制到可视canvas，并叠加选中/预览层
function drawFromBaseLayer() {
    const canvas = resultCanvas;
    if (!canvas || !baseRenderReady) return;
    const ctx = canvas.getContext('2d');
    // 清屏并设置当前视图变换
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.setTransform(viewScale, 0, 0, viewScale, viewOffsetX, viewOffsetY);
    ctx.drawImage(baseRenderCanvas, 0, 0);
    // 选中目标的高亮与句柄
    if (selectedDetectionIndex != null && detectionResults[selectedDetectionIndex] && Array.isArray(detectionResults[selectedDetectionIndex].polygon)) {
        const det = detectionResults[selectedDetectionIndex];
        const poly = det.polygon;
        const isValid = Array.isArray(poly) && poly.length > 2;
        if (isValid) {
            ctx.save();
            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#00E0FF';
            poly.forEach((pt, i) => {
                if (i === 0) ctx.moveTo(pt[0], pt[1]);
                else ctx.lineTo(pt[0], pt[1]);
            });
            ctx.closePath();
            ctx.stroke();
            // 句柄
            for (const [x, y] of poly) drawHandle(ctx, x, y);
            ctx.restore();
        }
    }
    // 复位
    ctx.setTransform(1, 0, 0, 1, 0, 0);
}

// 显示检测结果列表
function displayDetectionsList(detections) {
    detectionsList.innerHTML = '';

    detections.forEach((detection, index) => {
        const detectionItem = document.createElement('div');
        detectionItem.className = `detection-item fade-in ${index === selectedDetectionIndex ? 'selected' : ''} ${index === hoveredDetectionIndex ? 'hovered' : ''}`.trim();
        detectionItem.setAttribute('data-index', String(index));
        detectionItem.innerHTML = `
            <div class="d-flex justify-content-between align-items-start">
              <div class="detection-info" style="cursor: pointer;">
                  <div class="detection-label" data-index="${index}">${sanitizeLabel(detection.label)}</div>
                  <div class="detection-score">置信度: ${(detection.score * 100).toFixed(1)}%</div>
                  <div class="detection-box text-muted small">
                    [${detection.box.xmin}, ${detection.box.ymin}, ${detection.box.xmax}, ${detection.box.ymax}]
                  </div>
              </div>
              <div class="btn-group btn-group-sm" role="group">
                <button class="btn btn-outline-primary" data-action="select" data-index="${index}"><i class="fas fa-mouse-pointer"></i></button>
                <button class="btn btn-outline-secondary" data-action="edit-label" data-index="${index}"><i class="fas fa-pen"></i></button>
                <button class="btn btn-outline-danger" data-action="delete" data-index="${index}"><i class="fas fa-trash"></i></button>
              </div>
            </div>
        `;
        detectionsList.appendChild(detectionItem);
    });

    // 事件委托：选择/编辑标签/删除
    detectionsList.onclick = (e) => {
        const btn = e.target.closest('button');
        if (btn) {
            const action = btn.getAttribute('data-action');
            const idx = parseInt(btn.getAttribute('data-index'));
            if (Number.isInteger(idx)) {
                if (action === 'select') selectDetection(idx);
                if (action === 'edit-label') editLabelInline(idx);
                if (action === 'delete') deleteDetection(idx);
            }
            return;
        }
        const info = e.target.closest('.detection-info');
        if (info) {
            const idx = parseInt(info.querySelector('.detection-label').getAttribute('data-index'));
            if (Number.isInteger(idx)) selectDetection(idx);
        }
    };
    // 渲染后同步一次高亮（选中与悬浮）
    updateDetectionListSelection();
    updateDetectionListHover();
}

// 同步结果列表的选中高亮
function updateDetectionListSelection() {
    if (!detectionsList) return;
    const items = detectionsList.querySelectorAll('.detection-item');
    items.forEach(el => {
        const idx = parseInt(el.getAttribute('data-index'));
        if (Number.isInteger(idx)) {
            el.classList.toggle('selected', idx === selectedDetectionIndex);
        }
    });
}

// 清除结果
function clearResults() {
    currentImage = null;
    currentLabels = [];
    detectionResults = [];
    selectedDetectionIndex = null;
    draggingVertexIndex = -1;
    isDraggingVertex = false;

    // 清空顶部原图与结果画布
    if (originalImage) {
        originalImage.removeAttribute('src');
        originalImage.style.display = 'none';
    }
    if (originalPlaceholder) originalPlaceholder.style.display = '';
    if (resultCanvas) {
        const ctx = resultCanvas.getContext('2d');
        ctx && ctx.clearRect(0, 0, resultCanvas.width || 0, resultCanvas.height || 0);
        resultCanvas.width = 0;
        resultCanvas.height = 0;
    }

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

// ============ 交互与编辑 ============
function enableCanvasInteractions() {
    const canvas = resultCanvas;
    const getCanvasPos = (evt) => {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        // 先转到画布坐标，再反变换视图缩放和平移
        const cx = (evt.clientX - rect.left) * scaleX;
        const cy = (evt.clientY - rect.top) * scaleY;
        return {
            x: (cx - viewOffsetX) / viewScale,
            y: (cy - viewOffsetY) / viewScale
        };
    };

    canvas.onmousedown = (evt) => {
        const pos = getCanvasPos(evt);
        // 开始平移（按住空格 + 左键，或中键）
        if ((spacePressed && evt.button === 0) || evt.button === 1) {
            isPanning = true;
            lastPanClient = { x: evt.clientX, y: evt.clientY };
            return;
        }
        if (isAnnotating) {
            // Alt+点击顶点 -> 删除该点
            if (evt.altKey && annotationPoints.length > 0) {
                const vIdx = findNearbyVertex(annotationPoints, pos.x, pos.y, 8);
                if (vIdx !== -1) {
                    annotationPoints.splice(vIdx, 1);
                    redrawWithAnnotation();
                    return;
                }
            }
            // 添加一个顶点
            annotationPoints.push([Math.round(pos.x), Math.round(pos.y)]);
            redrawWithAnnotation();
            return;
        }

        // 若当前未选中任何目标，先尝试点击选中一个（支持直接点击选择mask/多边形/框）
        if (selectedDetectionIndex == null) {
            const hitIdx = findDetectionAtPoint(pos.x, pos.y);
            if (hitIdx !== -1) {
                // 直接设置索引以便本次事件后续逻辑可继续使用（避免需要再次点击）
                selectedDetectionIndex = hitIdx;
            } else {
                return; // 未点中任何目标
            }
        }
        const det = detectionResults[selectedDetectionIndex];
        if (!det || !Array.isArray(det.polygon)) return;

        // Ctrl + 点击边缘 -> 在该边插入新顶点并进入拖拽
        if (evt.ctrlKey) {
            const { edgeIndex, point } = findNearbyEdge(det.polygon, pos.x, pos.y, 8);
            if (edgeIndex !== -1 && point) {
                const insertAt = edgeIndex + 1;
                det.polygon.splice(insertAt, 0, point);
                updateBoxFromPolygon(det);
                baseRenderReady = false;
                buildBaseLayer(lastResultImageBase64, detectionResults).then(() => drawFromBaseLayer());
                isDraggingVertex = true;
                draggingVertexIndex = insertAt;
                return;
            }
        }

        // Alt+点击已存在多边形的顶点 -> 删除该点（编辑已有分割）
        if (evt.altKey) {
            const vIdx = findNearbyVertex(det.polygon, pos.x, pos.y, 8);
            if (vIdx !== -1) {
                if (det.polygon.length <= 3) {
                    showError('多边形至少需要3个点');
                    return;
                }
                det.polygon.splice(vIdx, 1);
                // 更新bbox
                updateBoxFromPolygon(det);
                // 变更需要重建底图
                baseRenderReady = false;
                buildBaseLayer(lastResultImageBase64, detectionResults).then(() => drawFromBaseLayer());
                return;
            }
        }

        // 先检测是否点中某个顶点
        const vIdx = findNearbyVertex(det.polygon, pos.x, pos.y, 8);
        if (vIdx !== -1) {
            isDraggingVertex = true;
            draggingVertexIndex = vIdx;
            return;
        }

        // 若未点中顶点，检测是否点击到多边形内部 -> 选中
        if (pointInPolygon(pos.x, pos.y, det.polygon)) {
            // 已选中则保持
        } else {
            // 尝试选择其它目标
            const idx = findDetectionAtPoint(pos.x, pos.y);
            if (idx !== -1) selectDetection(idx);
        }
    };

    canvas.onmousemove = (evt) => {
        if (isPanning) {
            const dx = evt.clientX - lastPanClient.x;
            const dy = evt.clientY - lastPanClient.y;
            lastPanClient = { x: evt.clientX, y: evt.clientY };
            // 平移以屏幕像素为基准 -> 转为画布像素（考虑当前canvas缩放到CSS显示的比例）
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            viewOffsetX += dx * scaleX;
            viewOffsetY += dy * scaleY;
            redrawWithAnnotation(evt, getCanvasPos);
            return;
        }
        if (isAnnotating) {
            // 预览最后一个点到鼠标位置的线段
            redrawWithAnnotation(evt, getCanvasPos);
            return;
        }
        const pos = getCanvasPos(evt);
        // 悬浮联动（未拖拽时，根据鼠标位置设置 hoveredDetectionIndex）
        if (!isDraggingVertex) {
            const idx = findDetectionAtPoint(pos.x, pos.y);
            setHoveredDetection(idx !== -1 ? idx : null);
        }
        if (!isDraggingVertex || selectedDetectionIndex == null) return;
        const det = detectionResults[selectedDetectionIndex];
        if (!det || !Array.isArray(det.polygon)) return;
        if (draggingVertexIndex >= 0 && draggingVertexIndex < det.polygon.length) {
            det.polygon[draggingVertexIndex] = [Math.round(pos.x), Math.round(pos.y)];
            redraw();
        }
    };

    const endDrag = () => {
        const wasDraggingVertex = isDraggingVertex;
        isDraggingVertex = false;
        draggingVertexIndex = -1;
        isPanning = false;
        // 拖拽顶点结束后，更新bbox并重建底图，避免底图中的旧轮廓与当前不一致
        if (wasDraggingVertex && selectedDetectionIndex != null) {
            const det = detectionResults[selectedDetectionIndex];
            if (det && Array.isArray(det.polygon)) {
                updateBoxFromPolygon(det);
                baseRenderReady = false;
                buildBaseLayer(lastResultImageBase64, detectionResults).then(() => {
                    drawFromBaseLayer();
                    // 保存修改后的结果
                    saveCurrentResultsToStorage();
                });
            }
        }
    };
    canvas.onmouseup = endDrag;
    canvas.onmouseleave = endDrag;

    // 缩放（滚轮缩放到光标处）——在任意模式下均可缩放，方便选择/编辑
    canvas.onwheel = (evt) => {
        evt.preventDefault();
        const delta = -Math.sign(evt.deltaY); // 上滚放大，下滚缩小
        const zoomFactor = 1 + (0.12 * delta);
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        const cx = (evt.clientX - rect.left) * scaleX;
        const cy = (evt.clientY - rect.top) * scaleY;
        const worldX = (cx - viewOffsetX) / viewScale;
        const worldY = (cy - viewOffsetY) / viewScale;
        // 应用缩放
        const newScale = Math.min(8, Math.max(0.2, viewScale * zoomFactor));
        // 保持鼠标处的世界坐标在屏幕位置不变：调整偏移
        viewOffsetX = cx - worldX * newScale;
        viewOffsetY = cy - worldY * newScale;
        viewScale = newScale;
        redrawWithAnnotation(evt, getCanvasPos);
    };

    // 双击完成标注
    canvas.ondblclick = () => {
        if (isAnnotating) finishAnnotation();
    };

    // 移出画布时清除悬浮高亮
    canvas.onmouseleave = () => {
        setHoveredDetection(null);
    };

    // 键盘事件：
    document.onkeydown = (e) => {
        // 在输入框/文本域/可编辑区域中不拦截任何快捷键，避免回退键影响多边形
        const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : '';
        const isEditable = (e.target && (e.target.isContentEditable || tag === 'input' || tag === 'textarea'));
        if (isEditable) return;
        // '?' 打开帮助（Shift + '/'）
        if (e.key === '?' || (e.shiftKey && e.key === '/')) {
            helpModal && helpModal.show();
            e.preventDefault();
            return;
        }
        if (isAnnotating) {
            if (e.key === 'Enter') {
                finishAnnotation();
            } else if (e.key === 'Escape') {
                cancelAnnotation();
            } else if (e.key === 'Backspace' || e.key === 'Delete') {
                if (annotationPoints.length > 0) {
                    annotationPoints.pop();
                    redrawWithAnnotation();
                }
                e.preventDefault();
            } else if (e.code === 'Space') {
                spacePressed = true;
                e.preventDefault();
            }
            return;
        }
        // 非标注模式：若选中已有多边形，支持 Backspace/Delete 删除最后一个点
        if (selectedDetectionIndex != null && (e.key === 'Backspace' || e.key === 'Delete')) {
            const det = detectionResults[selectedDetectionIndex];
            if (det && Array.isArray(det.polygon) && det.polygon.length > 3) {
                det.polygon.pop();
                updateBoxFromPolygon(det);
                baseRenderReady = false;
                buildBaseLayer(lastResultImageBase64, detectionResults).then(() => drawFromBaseLayer());
            } else {
                showError('多边形至少需要3个点');
            }
            e.preventDefault();
        } else if (e.code === 'Space') {
            spacePressed = true;
            e.preventDefault();
        }
    };
    document.onkeyup = (e) => {
        if (e.code === 'Space') spacePressed = false;
    };
}

function redraw() {
    if (!lastResultImageBase64) return;
    drawFromBaseLayer();
    // 在底图之上绘制悬浮高亮
    if (hoveredDetectionIndex != null) {
        const det = detectionResults[hoveredDetectionIndex];
        if (det && Array.isArray(det.polygon) && det.polygon.length > 2) {
            const ctx = resultCanvas.getContext('2d');
            if (ctx) {
                ctx.setTransform(viewScale, 0, 0, viewScale, viewOffsetX, viewOffsetY);
                ctx.save();
                ctx.beginPath();
                det.polygon.forEach((pt, i) => {
                    if (i === 0) ctx.moveTo(pt[0], pt[1]); else ctx.lineTo(pt[0], pt[1]);
                });
                ctx.closePath();
                ctx.fillStyle = 'rgba(79,70,229,0.10)';
                ctx.strokeStyle = '#4f46e5';
                ctx.lineWidth = 3;
                ctx.fill();
                ctx.stroke();
                ctx.restore();
                ctx.setTransform(1, 0, 0, 1, 0, 0);
            }
        }
    }
}

function redrawWithAnnotation(evt, getCanvasPos) {
    drawFromBaseLayer();
    const ctx = resultCanvas.getContext('2d');
    if (!ctx || annotationPoints.length === 0) return;
    // 应用当前视图变换，在其上预览
    ctx.setTransform(viewScale, 0, 0, viewScale, viewOffsetX, viewOffsetY);
    ctx.save();
    ctx.strokeStyle = '#FFA500';
    ctx.fillStyle = 'rgba(255,165,0,0.15)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < annotationPoints.length; i++) {
        const [x, y] = annotationPoints[i];
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    // 临时预览线段到当前鼠标
    if (evt && typeof getCanvasPos === 'function') {
        const p = getCanvasPos(evt);
        const last = annotationPoints[annotationPoints.length - 1];
        ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    // 画点句柄
    for (const [x, y] of annotationPoints) drawHandle(ctx, x, y);
    ctx.restore();
    // 复位
    ctx.setTransform(1, 0, 0, 1, 0, 0);
}

function toggleAnnotationMode() {
    if (!resultCanvas || !resultCanvas.width) {
        showError('请先运行分割或加载结果图像后再进行标注');
        return;
    }
    isAnnotating = !isAnnotating;
    annotationPoints = [];
    annotateToggleBtn.classList.toggle('active', isAnnotating);
    annotateToggleBtn.innerHTML = isAnnotating
        ? '<i class="fas fa-draw-polygon"></i> 退出标注模式'
        : '<i class="fas fa-draw-polygon"></i> 手动标注模式';
    // 进入标注模式时重置视图到居中1x；退出时也复位
    viewScale = 1;
    viewOffsetX = 0;
    viewOffsetY = 0;
    redraw();
}

function finishAnnotation() {
    if (annotationPoints.length < 3) {
        showError('多边形至少需要3个点');
        return;
    }
    // 询问标签
    const label = prompt('输入该标注的标签：', (currentLabels[0] || 'custom')) || 'custom';
    // 计算bbox
    const xs = annotationPoints.map(p => p[0]);
    const ys = annotationPoints.map(p => p[1]);
    const box = {
        xmin: Math.min(...xs),
        ymin: Math.min(...ys),
        xmax: Math.max(...xs),
        ymax: Math.max(...ys)
    };
    const det = { label, score: 1.0, box, polygon: [...annotationPoints], is_manual: true };
    detectionResults.push(det);
    displayDetectionsList(detectionResults);
    // 结束标注
    isAnnotating = false;
    annotationPoints = [];
    annotateToggleBtn.classList.remove('active');
    annotateToggleBtn.innerHTML = '<i class="fas fa-draw-polygon"></i> 手动标注模式';
    // 新增的手动标注需要体现在底图
    baseRenderReady = false;
    buildBaseLayer(lastResultImageBase64, detectionResults).then(() => {
        drawFromBaseLayer();
        // 保存修改后的结果
        saveCurrentResultsToStorage();
    });
}

function cancelAnnotation() {
    annotationPoints = [];
    isAnnotating = false;
    annotateToggleBtn.classList.remove('active');
    annotateToggleBtn.innerHTML = '<i class="fas fa-draw-polygon"></i> 手动标注模式';
    redraw();
}

function drawHandle(ctx, x, y) {
    ctx.save();
    ctx.beginPath();
    ctx.fillStyle = '#00E0FF';
    ctx.strokeStyle = '#004D66';
    ctx.lineWidth = 1.5;
    ctx.arc(x, y, 4.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.restore();
}

function findNearbyVertex(poly, x, y, tol = 6) {
    for (let i = 0; i < poly.length; i++) {
        const [px, py] = poly[i];
        if (Math.hypot(px - x, py - y) <= tol) return i;
    }
    return -1;
}

// 在多边形边缘附近查找可插入的新顶点位置
function closestPointOnSegment(x1, y1, x2, y2, px, py) {
    const vx = x2 - x1, vy = y2 - y1;
    const wx = px - x1, wy = py - y1;
    const len2 = vx * vx + vy * vy || 1e-9;
    let t = (vx * wx + vy * wy) / len2;
    t = Math.max(0, Math.min(1, t));
    return { x: x1 + t * vx, y: y1 + t * vy };
}

function findNearbyEdge(poly, x, y, tol = 8) {
    let best = { edgeIndex: -1, point: null, dist: Infinity };
    for (let i = 0; i < poly.length; i++) {
        const a = poly[i];
        const b = poly[(i + 1) % poly.length];
        const cp = closestPointOnSegment(a[0], a[1], b[0], b[1], x, y);
        const d = Math.hypot(cp.x - x, cp.y - y);
        if (d < best.dist) {
            best = { edgeIndex: i, point: [Math.round(cp.x), Math.round(cp.y)], dist: d };
        }
    }
    if (best.dist <= tol) return { edgeIndex: best.edgeIndex, point: best.point };
    return { edgeIndex: -1, point: null };
}

function pointInPolygon(x, y, polygon) {
    // ray casting
    let inside = false;
    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
        const xi = polygon[i][0], yi = polygon[i][1];
        const xj = polygon[j][0], yj = polygon[j][1];
        const intersect = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi + 1e-9) + xi);
        if (intersect) inside = !inside;
    }
    return inside;
}

// 根据多边形更新bbox
function updateBoxFromPolygon(det) {
    if (!det || !Array.isArray(det.polygon) || det.polygon.length < 3) return;
    const xs = det.polygon.map(p => p[0]);
    const ys = det.polygon.map(p => p[1]);
    det.box = {
        xmin: Math.min(...xs),
        ymin: Math.min(...ys),
        xmax: Math.max(...xs),
        ymax: Math.max(...ys)
    };
}

function findDetectionAtPoint(x, y) {
    // 优先查找包含点的多边形
    for (let i = detectionResults.length - 1; i >= 0; i--) {
        const det = detectionResults[i];
        if (Array.isArray(det.polygon) && det.polygon.length > 2) {
            if (pointInPolygon(x, y, det.polygon)) return i;
        } else {
            // 回退到bbox
            const b = det.box;
            if (x >= b.xmin && x <= b.xmax && y >= b.ymin && y <= b.ymax) return i;
        }
    }
    return -1;
}

function selectDetection(index) {
    if (index < 0 || index >= detectionResults.length) return;
    selectedDetectionIndex = index;
    redraw();
    updateDetectionListSelection();
}

function editLabelInline(index) {
    if (index < 0 || index >= detectionResults.length) return;
    // 找到对应DOM元素并替换为输入框
    const labelDivs = detectionsList.querySelectorAll('.detection-label');
    const labelDiv = Array.from(labelDivs).find(el => parseInt(el.getAttribute('data-index')) === index);
    if (!labelDiv) return;
    const old = detectionResults[index].label;
    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-control form-control-sm';
    input.value = old;
    labelDiv.replaceWith(input);
    input.focus();
    const commit = () => {
        const v = input.value.trim();
        if (v) detectionResults[index].label = sanitizeLabel(v);
        displayDetectionsList(detectionResults);
        // 标签变更影响底图文字，重建底图
        baseRenderReady = false;
        buildBaseLayer(lastResultImageBase64, detectionResults).then(() => drawFromBaseLayer());
        // 保存修改后的结果
        saveCurrentResultsToStorage();
    };
    input.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
    input.addEventListener('blur', commit);
}

function deleteDetection(index) {
    if (index < 0 || index >= detectionResults.length) return;
    detectionResults.splice(index, 1);
    if (selectedDetectionIndex === index) selectedDetectionIndex = null;
    if (selectedDetectionIndex > index) selectedDetectionIndex--;
    displayDetectionsList(detectionResults);
    baseRenderReady = false;
    buildBaseLayer(lastResultImageBase64, detectionResults).then(() => drawFromBaseLayer());
    
    // 保存修改后的结果
    saveCurrentResultsToStorage();
}

// localStorage相关工具函数
function getStorageKeyForImage(relPath) {
    return `${STORAGE_KEY_PREFIX}${relPath}`;
}

function saveResultsToLocalStorage(relPath, resultData) {
    if (!ENABLE_LOCAL_STORAGE || !relPath) return;
    try {
        const storageKey = getStorageKeyForImage(relPath);
        localStorage.setItem(storageKey, JSON.stringify(resultData));
    } catch (err) {
        console.error('Failed to save results to localStorage:', err);
        // 如果存储失败（可能是存储空间已满），尝试清理一些旧数据
        if (err.name === 'QuotaExceededError') {
            cleanupLocalStorage();
        }
    }
}

function saveCurrentResultsToStorage() {
    if (!ENABLE_LOCAL_STORAGE || currentImageIndex < 0) return;
    const item = imageQueue[currentImageIndex];
    if (!item || !item.relPath) return;
    
    const resultData = {
        detections: JSON.parse(JSON.stringify(detectionResults)),
        resultImageBase64: lastResultImageBase64
    };
    
    // 保存到内存
    perImageResults.set(currentImageIndex, resultData);
    
    // 保存到localStorage
    saveResultsToLocalStorage(item.relPath, resultData);
}

function cleanupLocalStorage() {
    // 简单策略：移除最旧的一半数据
    try {
        const keys = [];
        for (let i = 0; i < localStorage.length; i++) {
            const key = localStorage.key(i);
            if (key.startsWith(STORAGE_KEY_PREFIX)) {
                keys.push(key);
            }
        }
        
        if (keys.length > 10) { // 只有当有足够多的项时才清理
            // 按字母顺序排序，简单处理
            keys.sort();
            // 移除前一半
            const removeCount = Math.floor(keys.length / 2);
            for (let i = 0; i < removeCount; i++) {
                localStorage.removeItem(keys[i]);
            }
            console.log(`Cleaned up ${removeCount} old items from localStorage`);
        }
    } catch (err) {
        console.error('Failed to cleanup localStorage:', err);
    }
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
