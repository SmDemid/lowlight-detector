// ---------- Глобальные переменные ----------
let progressInterval = null;

// ---------- Вспомогательные функции ----------
function showLoading(message = 'Обработка...') {
    document.getElementById('loadingPanel').classList.remove('d-none');
    document.getElementById('loadingMessage').textContent = message;
    document.getElementById('progressBar').style.width = '0%';
    
    let width = 0;
    progressInterval = setInterval(() => {
        if (width >= 90) {
            clearInterval(progressInterval);
        } else {
            width += 5;
            document.getElementById('progressBar').style.width = width + '%';
        }
    }, 200);
}

function hideLoading() {
    document.getElementById('loadingPanel').classList.add('d-none');
    if (progressInterval) {
        clearInterval(progressInterval);
        progressInterval = null;
    }
    document.getElementById('progressBar').style.width = '100%';
}

function getSelectedEnhancers() {
    const enhancers = [];
    document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => enhancers.push(cb.value));
    return enhancers;
}

function appendCommonParams(formData) {
    formData.append('detector', document.getElementById('detectorSelect').value);
    formData.append('conf_threshold', document.getElementById('confThreshold').value);
    const enhancers = getSelectedEnhancers();
    enhancers.forEach(e => formData.append('enhancers', e));
}

// ---------- Отображение списка выбранных файлов ----------
document.getElementById('imageUpload').addEventListener('change', function(e) {
    const files = Array.from(e.target.files);
    const listDiv = document.getElementById('fileList');
    
    if (files.length === 0) {
        listDiv.innerHTML = '';
        return;
    }
    
    let html = `<strong>Выбрано файлов: ${files.length}</strong><ul class="list-unstyled mt-1">`;
    files.forEach((f, i) => {
        if (i < 5) {
            html += `<li>${f.name}</li>`;
        } else if (i === 5) {
            html += `<li>... и ещё ${files.length - 5}</li>`;
        }
    });
    html += '</ul>';
    listDiv.innerHTML = html;
});

// ---------- Обновление отображения порога уверенности ----------
document.getElementById('confThreshold').addEventListener('input', function() {
    document.getElementById('confValue').textContent = this.value;
});

// ---------- Функция добавления карточки результата ----------
function addResultCard(container, title, result) {
    const col = document.createElement('div');
    col.className = 'col-md-6 col-lg-4';
    
    const card = document.createElement('div');
    card.className = 'card h-100';
    
    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${result.annotated_base64}`;
    img.className = 'card-img-top img-card';
    img.alt = title;
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    
    const count = result.count || 0;
    const avgConf = result.avg_confidence ? (result.avg_confidence * 100).toFixed(1) : '0.0';
    const detTime = result.detection_time_ms ? result.detection_time_ms.toFixed(0) : '—';
    const enhTime = result.enhancement_time_ms ? result.enhancement_time_ms.toFixed(0) : null;
    
    let metricsHtml = `
        <h5 class="card-title">${title}</h5>
        <p class="mb-1">Обнаружено объектов: <strong>${count}</strong></p>
        <p class="mb-1">Средняя уверенность: ${avgConf}%</p>
        <p class="mb-1">Время детекции: ${detTime} мс</p>
    `;
    
    if (enhTime) {
        metricsHtml += `<p class="mb-1">Время улучшения: ${enhTime} мс</p>`;
    }
    
    cardBody.innerHTML = metricsHtml;
    
    // Кнопки переключения между исходным и аннотированным изображением
    if (result.original_base64) {
        const btnGroup = document.createElement('div');
        btnGroup.className = 'btn-group btn-group-sm mt-2';
        btnGroup.innerHTML = `
            <button class="btn btn-outline-secondary active" data-view="annotated">С детекцией</button>
            <button class="btn btn-outline-secondary" data-view="original">Исходное</button>
        `;
        
        btnGroup.querySelectorAll('button').forEach(btn => {
            btn.addEventListener('click', (e) => {
                btnGroup.querySelectorAll('button').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                if (btn.dataset.view === 'original') {
                    img.src = `data:image/jpeg;base64,${result.original_base64}`;
                } else {
                    img.src = `data:image/jpeg;base64,${result.annotated_base64}`;
                }
            });
        });
        
        cardBody.appendChild(btnGroup);
    }
    
    card.appendChild(img);
    card.appendChild(cardBody);
    col.appendChild(card);
    container.appendChild(col);
}

// ---------- Обработка одного изображения ----------
function displaySingleResults(data) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    addResultCard(container, 'Оригинал', data.original);
    
    for (const [method, result] of Object.entries(data.enhanced)) {
        addResultCard(container, method, result);
    }
}

document.getElementById('analyzeSingleBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
        alert('Выберите изображение!');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);
    appendCommonParams(formData);

    showLoading('Обработка одного изображения...');
    
    try {
        const response = await fetch('/api/process', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        displaySingleResults(data);
    } catch (error) {
        alert('Ошибка: ' + error.message);
        console.error(error);
    } finally {
        hideLoading();
    }
});

// ---------- Загрузка детальной информации по одному изображению из пакета ----------
async function loadSingleImageDetail(batchId, index) {
    const detailContainer = document.getElementById('singleImageDetail');
    detailContainer.innerHTML = `
        <div class="text-center">
            <div class="spinner-border text-primary"></div> 
            Загрузка детекций...
        </div>
    `;
    
    try {
        const response = await fetch(`/api/batch-image/${batchId}/${index}`);
        if (!response.ok) throw new Error('Ошибка загрузки');
        const data = await response.json();
        
        detailContainer.innerHTML = `
            <h5 class="mt-3">${data.image_name}</h5>
            <div class="row" id="detailCards"></div>
        `;
        
        const cardsContainer = document.getElementById('detailCards');
        addResultCard(cardsContainer, 'Оригинал', data.original);
        
        for (const [method, res] of Object.entries(data.enhanced)) {
            addResultCard(cardsContainer, method, res);
        }
    } catch (error) {
        detailContainer.innerHTML = `<div class="alert alert-danger">Ошибка: ${error.message}</div>`;
    }
}

// ---------- Пакетная обработка ----------
function displayBatchStatistics(data) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    const stats = data.aggregated_stats;
    const totalImages = data.total_images || 0;
    const individualResults = data.individual_results || [];
    const batchId = data.batch_id;
    
    // Сводная таблица
    let summaryRows = '';
    const orig = stats.original;
    summaryRows += `<tr><td><strong>Оригинал</strong></td><td>${orig.avg_count.toFixed(2)}</td><td>${(orig.avg_confidence * 100).toFixed(1)}%</td><td>${orig.avg_detection_time_ms.toFixed(0)}</td><td>-</td></tr>`;
    
    for (const [method, s] of Object.entries(stats.enhanced)) {
        const gain = s.count_gain || 0;
        const gainPercent = s.count_gain_percent || 0;
        const gainStr = `${gain > 0 ? '+' : ''}${gain.toFixed(2)} (${gainPercent.toFixed(1)}%)`;
        summaryRows += `<tr><td><strong>${method}</strong></td><td>${s.avg_count.toFixed(2)}</td><td>${(s.avg_confidence * 100).toFixed(1)}%</td><td>${s.avg_total_time_ms.toFixed(0)}</td><td>${gainStr}</td></tr>`;
    }
    
    // Индивидуальные результаты
    let individualHtml = `
        <h5 class="mt-4">Результаты по каждому изображению (кликните для просмотра детекций)</h5>
        <div class="table-responsive" style="max-height: 400px; overflow-y: auto;">
            <table class="table table-striped table-hover">
                <thead>
                    <tr>
                        <th>Изображение</th>
                        <th>Метод</th>
                        <th>Объектов</th>
                        <th>Уверенность</th>
                        <th>Время (мс)</th>
                    </tr>
                </thead>
                <tbody>
    `;
    
    individualResults.forEach((result, idx) => {
        const imgName = result.image_name || '—';
        const origCount = result.original.count || 0;
        const origConf = result.original.avg_confidence ? (result.original.avg_confidence * 100).toFixed(1) : '0.0';
        const origTime = result.original.detection_time_ms ? result.original.detection_time_ms.toFixed(0) : '—';
        
        individualHtml += `<tr class="batch-image-row" data-batch-id="${batchId}" data-index="${idx}">`;
        individualHtml += `<td rowspan="${Object.keys(result.enhanced).length + 1}"><strong>${imgName}</strong></td>`;
        individualHtml += `<td>Оригинал</td><td>${origCount}</td><td>${origConf}%</td><td>${origTime}</td></tr>`;
        
        for (const [method, res] of Object.entries(result.enhanced)) {
            const count = res.count || 0;
            const conf = res.avg_confidence ? (res.avg_confidence * 100).toFixed(1) : '0.0';
            const time = res.total_time_ms ? res.total_time_ms.toFixed(0) : '—';
            individualHtml += `<tr class="batch-image-row" data-batch-id="${batchId}" data-index="${idx}">`;
            individualHtml += `<td>${method}</td><td>${count}</td><td>${conf}%</td><td>${time}</td></tr>`;
        }
    });
    
    individualHtml += `</tbody></table></div>`;
    individualHtml += `<div id="singleImageDetail" class="mt-4"></div>`;
    
    // Сборка всего интерфейса
    const col = document.createElement('div');
    col.className = 'col-12';
    col.innerHTML = `
        <div class="card mb-3">
            <div class="card-header">
                <h4>Сводная статистика (${totalImages} изображений)</h4>
            </div>
            <div class="card-body">
                <div class="table-responsive">
                    <table class="table table-striped">
                        <thead>
                            <tr>
                                <th>Метод</th>
                                <th>Среднее объектов</th>
                                <th>Средняя уверенность</th>
                                <th>Среднее время (мс)</th>
                                <th>Прирост объектов</th>
                            </tr>
                        </thead>
                        <tbody>${summaryRows}</tbody>
                    </table>
                </div>
                <canvas id="batchChart" width="400" height="200" class="mt-4"></canvas>
            </div>
        </div>
        <div class="card">
            <div class="card-header">
                <h4>Детальные результаты</h4>
            </div>
            <div class="card-body">
                ${individualHtml}
            </div>
        </div>
    `;
    
    container.appendChild(col);
    
    // Построение графика
    const ctx = document.getElementById('batchChart').getContext('2d');
    const labels = ['Оригинал', ...Object.keys(stats.enhanced)];
    const counts = [stats.original.avg_count, ...Object.values(stats.enhanced).map(s => s.avg_count)];
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Среднее количество обнаруженных объектов',
                data: counts,
                backgroundColor: ['#6c757d', '#0d6efd', '#198754', '#ffc107', '#dc3545', '#0dcaf0']
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: { 
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Количество объектов'
                    }
                }
            }
        }
    });
    
    // Добавление обработчиков кликов для загрузки деталей
    document.querySelectorAll('.batch-image-row').forEach(row => {
        row.addEventListener('click', async (e) => {
            const clickedBatchId = row.dataset.batchId;
            const index = row.dataset.index;
            await loadSingleImageDetail(clickedBatchId, index);
        });
    });
}

document.getElementById('analyzeBatchBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
        alert('Выберите хотя бы одно изображение!');
        return;
    }
    
    const files = Array.from(fileInput.files);
    const formData = new FormData();
    files.forEach(file => formData.append('images', file));
    appendCommonParams(formData);

    showLoading(`Обработка пакета из ${files.length} изображений...`);
    
    try {
        const response = await fetch('/api/analyze-batch', { method: 'POST', body: formData });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        displayBatchStatistics(data);
    } catch (error) {
        alert('Ошибка: ' + error.message);
        console.error(error);
    } finally {
        hideLoading();
    }
});

// ---------- Инициализация при загрузке страницы ----------
window.onload = function() {
    document.getElementById('confValue').textContent = document.getElementById('confThreshold').value;
};