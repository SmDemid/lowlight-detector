import os
import tempfile
import shutil
import uuid
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
from werkzeug.utils import secure_filename

from .enhancement.clahe import CLAHEEnhancer
from .enhancement.gamma import GammaEnhancer
from .enhancement.msrcr import MSRCREnhancer
from .enhancement.bilateral import BilateralEnhancer
from .enhancement.zero_dce import ZeroDCEEnhancer

from .detection.yolo_detector import YOLODetector
from .detection.rcnn_detector import FasterRCNNDetector

from .analysis import ModelAnalyzer

from .utils.image_utils import load_image_from_file, image_to_base64, resize_image


batch_cache = {}

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 МБ

# Инициализация компонентов
ENHANCERS = {
    'clahe': CLAHEEnhancer(),
    'gamma': GammaEnhancer(),
    'msrcr': MSRCREnhancer(),
    'bilateral': BilateralEnhancer(),
    'zero_dce': ZeroDCEEnhancer(model_path='models/zero_dce.pth')
}

DETECTORS = {
    'yolov8n': YOLODetector('n'),
    'yolov8m': YOLODetector('m'),
    'faster_rcnn': FasterRCNNDetector()
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'enhancers': list(ENHANCERS.keys()),
        'detectors': list(DETECTORS.keys())
    })

@app.route('/api/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        image = load_image_from_file(file)
        image = resize_image(image, max_size=1024)  # ограничение размера
    except Exception as e:
        return jsonify({'error': f'Invalid image: {str(e)}'}), 400
    
    # Параметры запроса
    enhancer_names = request.form.getlist('enhancers')
    if not enhancer_names:
        enhancer_names = ['clahe']  # по умолчанию
    
    detector_name = request.form.get('detector', 'yolov8m')
    conf_threshold = float(request.form.get('conf_threshold', 0.25))
    
    if detector_name not in DETECTORS:
        return jsonify({'error': f'Unknown detector: {detector_name}'}), 400
    
    detector = DETECTORS[detector_name]
    detector.confidence_threshold = conf_threshold
    
    result = {
        'original': {},
        'enhanced': {}
    }
    
    # 1. Обработка исходного изображения
    orig_det = detector.process(image)
    result['original'] = {
        'image_base64': image_to_base64(image),
        'annotated_base64': image_to_base64(orig_det['annotated_image']),
        'detections': orig_det['detections'],
        'count': orig_det['count'],
        'avg_confidence': orig_det['avg_confidence'],
        'detection_time_ms': orig_det['time_ms']
    }
    
    # 2. Обработка с улучшениями
    for name in enhancer_names:
        if name not in ENHANCERS:
            continue
        
        enhancer = ENHANCERS[name]
        enh_result = enhancer.process(image)
        det_result = detector.process(enh_result['image'])
        
        result['enhanced'][name] = {
            'image_base64': image_to_base64(enh_result['image']),
            'annotated_base64': image_to_base64(det_result['annotated_image']),
            'detections': det_result['detections'],
            'count': det_result['count'],
            'avg_confidence': det_result['avg_confidence'],
            'enhancement_time_ms': enh_result['metrics']['time_ms'],
            'detection_time_ms': det_result['time_ms'],
            'metrics': enh_result['metrics']
        }
    
    return jsonify(result)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    if not data or 'path' not in data:
        return jsonify({'error': 'Path is required'}), 400

    path = data['path']
    detector_name = data.get('detector', 'yolov8m')
    conf_threshold = float(data.get('confidence_threshold', 0.25))
    enhancers = data.get('enhancers')
    include_images = data.get('include_images', False)

    analyzer = ModelAnalyzer(detector_name, conf_threshold)

    if os.path.isdir(path):
        result = analyzer.analyze_folder(
            folder_path=path,
            enhancer_names=enhancers,
            include_images=include_images
        )
    elif os.path.isfile(path):
        result = analyzer.analyze_single_image(
            image_path=path,
            enhancer_names=enhancers,
            include_images=include_images
        )
    else:
        return jsonify({'error': 'Path does not exist'}), 400

    return jsonify(result)

@app.route('/api/analyze-batch', methods=['POST'])
def analyze_batch():
    """Обработка нескольких изображений за один запрос."""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400

    files = request.files.getlist('images')
    if not files:
        return jsonify({'error': 'Empty file list'}), 400

    # Параметры
    detector_name = request.form.get('detector', 'yolov8m')
    conf_threshold = float(request.form.get('conf_threshold', 0.25))
    enhancer_names = request.form.getlist('enhancers')
    include_individual = request.form.get('include_individual', 'true').lower() == 'true'

    # Создаём временную папку
    temp_dir = tempfile.mkdtemp()
    saved_paths = []

    try:
        # Сохраняем все файлы
        for file in files:
            if file and file.filename:
                filename = secure_filename(file.filename)
                filepath = os.path.join(temp_dir, filename)
                file.save(filepath)
                saved_paths.append(filepath)

        if not saved_paths:
            return jsonify({'error': 'No valid files saved'}), 400

        # Импортируем ModelAnalyzer (если не импортирован глобально)
        from .analysis import ModelAnalyzer
        
        # Создаём экземпляр анализатора
        analyzer = ModelAnalyzer(detector_name, conf_threshold)
        
        # Генерируем ID пакета
        batch_id = str(uuid.uuid4())
        
        # Получаем результаты С ИЗОБРАЖЕНИЯМИ для кэша
        result_with_images = analyzer.analyze_folder(
            folder_path=temp_dir,
            enhancer_names=enhancer_names,
            include_images=True,           # важно для кэша
            include_individual=True
        )
        
        # Сохраняем в кэш
        batch_cache[batch_id] = result_with_images
        
        # Для ответа клиенту убираем base64 (экономия трафика)
        result_light = {
            'batch_id': batch_id,
            'total_images': result_with_images['total_images'],
            'detector': result_with_images['detector'],
            'confidence_threshold': result_with_images['confidence_threshold'],
            'aggregated_stats': result_with_images['aggregated_stats'],
            'individual_results': []
        }
        
        # Копируем индивидуальные результаты без изображений
        if include_individual and 'individual_results' in result_with_images:
            for res in result_with_images['individual_results']:
                result_light['individual_results'].append(
                    analyzer._strip_images_from_result(res)
                )
        
        return jsonify(result_light)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

    finally:
        # Удаляем временную папку
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/api/batch-image/<batch_id>/<int:index>', methods=['GET'])
def get_batch_image(batch_id, index):
    """Возвращает детальные результаты (с base64) для одного изображения из кэша пакета."""
    if batch_id not in batch_cache:
        return jsonify({'error': 'Batch not found or expired'}), 404
    
    batch = batch_cache[batch_id]
    if 'individual_results' not in batch or index >= len(batch['individual_results']):
        return jsonify({'error': 'Invalid image index'}), 400
    
    return jsonify(batch['individual_results'][index])


@app.route('/api/compare_models', methods=['POST'])
def compare_models():
    """
    Сравнение двух моделей (например, yolov8n vs yolov8m) на одном изображении или папке.
    """
    data = request.get_json()
    # ... реализация по аналогии
    pass

if __name__ == '__main__':
    # Создаём папку models, если её нет
    os.makedirs('models', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)