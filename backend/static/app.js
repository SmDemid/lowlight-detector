document.getElementById('analyzeBtn').addEventListener('click', async () => {
    const fileInput = document.getElementById('imageUpload');
    if (!fileInput.files.length) {
        alert('Выберите изображение!');
        return;
    }
    
    const file = fileInput.files[0];
    const formData = new FormData();
    formData.append('image', file);
    
    // Собираем параметры
    const detector = document.getElementById('detectorSelect').value;
    const confThreshold = document.getElementById('confThreshold').value;
    const enhancers = [];
    document.querySelectorAll('input[type="checkbox"]:checked').forEach(cb => enhancers.push(cb.value));
    
    // Отправляем на сервер (предположим, используем существующий /api/process)
    const response = await fetch('/api/process', {
        method: 'POST',
        body: formData
    });
    
    const data = await response.json();
    displayResults(data);
});

function displayResults(data) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    // Карточка для оригинала
    addResultCard(container, 'Оригинал', data.original);
    
    // Карточки для улучшенных
    for (const [method, result] of Object.entries(data.enhanced)) {
        addResultCard(container, method, result);
    }
}

function addResultCard(container, title, result) {
    const col = document.createElement('div');
    col.className = 'col-md-4';
    
    const card = document.createElement('div');
    card.className = 'card bg-secondary text-light';
    
    // Изображение (аннотированное)
    const img = document.createElement('img');
    img.src = `data:image/jpeg;base64,${result.annotated_base64}`;
    img.className = 'card-img-top img-card';
    
    const cardBody = document.createElement('div');
    cardBody.className = 'card-body';
    cardBody.innerHTML = `
        <h5 class="card-title">${title}</h5>
        <p>Обнаружено объектов: <strong>${result.count}</strong></p>
        <p>Средняя уверенность: ${(result.avg_confidence * 100).toFixed(1)}%</p>
        <p>Время детекции: ${result.detection_time_ms?.toFixed(0) || '—'} мс</p>
    `;
    
    card.appendChild(img);
    card.appendChild(cardBody);
    col.appendChild(card);
    container.appendChild(col);
}