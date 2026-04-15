import os
import cv2
import numpy as np
import base64
from typing import List, Dict, Any, Optional

from .enhancement import get_enhancer
from .detection import get_detector


def _image_to_base64(image: np.ndarray, quality: int = 85) -> str:
    """Конвертирует numpy-изображение (BGR) в base64 строку JPEG."""
    _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buffer).decode('utf-8')


class ModelAnalyzer:
    """Класс для сравнительного анализа детекции до и после улучшения."""
    
    def __init__(self, detector_name: str = 'yolov8m', confidence_threshold: float = 0.25):
        self.detector_name = detector_name
        self.confidence_threshold = confidence_threshold
        self.detector = get_detector(detector_name, confidence_threshold)
        
        # Инициализируем все доступные методы улучшения
        self.enhancers = {
            'clahe': get_enhancer('clahe'),
            'gamma': get_enhancer('gamma'),
            'msrcr': get_enhancer('msrcr'),
            'bilateral': get_enhancer('bilateral'),
            'zero_dce': get_enhancer('zero_dce')
        }
    
    def analyze_single_image(self, image_path: str, enhancer_names: Optional[List[str]] = None, include_images: bool = True ) -> Dict[str, Any]:
        """
        Анализ одного изображения.
        Возвращает словарь с результатами до и после улучшения.
        """
        if enhancer_names is None:
            enhancer_names = list(self.enhancers.keys())
        
        image = cv2.imread(image_path)
        if image is None:
            return {'error': f'Could not read image: {image_path}'}
        
        results = {
            'image_name': os.path.basename(image_path),
            'original': self._process_original(image),
            'enhanced': {}
        }
        
        for name in enhancer_names:
            if name in self.enhancers:
                enhancer = self.enhancers[name]
                results['enhanced'][name] = self._process_enhanced(image, enhancer)
        
        return results
    
    def analyze_folder(self, folder_path: str, enhancer_names: Optional[List[str]] = None, include_images: bool = False) -> Dict[str, Any]:
        """
        Анализ всех изображений в папке и агрегация статистики.
        """
        if enhancer_names is None:
            enhancer_names = list(self.enhancers.keys())
        
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        image_files = [f for f in os.listdir(folder_path) 
                       if f.lower().endswith(image_extensions)]
        
        if not image_files:
            return {'error': f'No images found in {folder_path}'}
        
        all_results = []
        aggregated = self._init_aggregated_stats(enhancer_names)
        
        for img_file in image_files:
            img_path = os.path.join(folder_path, img_file)
            result = self.analyze_single_image(
                image_path=img_path,
                enhancer_names=enhancer_names,
                include_images=include_images
            )
            if 'error' not in result:
                all_results.append(result)
                self._update_aggregated_stats(aggregated, result)
        
        # Вычисляем средние значения
        num_images = len(all_results)
        aggregated = self._finalize_aggregated_stats(aggregated, num_images)
        
        return {
            'total_images': num_images,
            'detector': self.detector_name,
            'confidence_threshold': self.confidence_threshold,
            'aggregated_stats': aggregated,
            'per_image_results': all_results  # можно отключить для больших папок
        }
    
    def _process_original(self, image: np.ndarray, include_images: bool) -> Dict[str, Any]:
        det_result = self.detector.process(image)
        annotated = self.detector._draw_detections(image.copy(), det_result['detections'])

        result = {
            'detections': det_result['detections'],
            'count': det_result['count'],
            'avg_confidence': det_result['avg_confidence'],
            'detection_time_ms': det_result['time_ms'],
            'metrics': {
                'brightness': self._compute_brightness(image),
                'contrast': self._compute_contrast(image)
            }
        }

        if include_images:
            result['annotated_base64'] = _image_to_base64(annotated)
            result['original_base64'] = _image_to_base64(image)

        return result
    
    def _process_enhanced(
        self,
        image: np.ndarray,
        enhancer,
        include_images: bool
    ) -> Dict[str, Any]:
        enh_result = enhancer.process(image)
        enhanced_img = enh_result['image']
        det_result = self.detector.process(enhanced_img)
        annotated = self.detector._draw_detections(enhanced_img.copy(), det_result['detections'])

        result = {
            'detections': det_result['detections'],
            'count': det_result['count'],
            'avg_confidence': det_result['avg_confidence'],
            'enhancement_time_ms': enh_result['metrics']['time_ms'],
            'detection_time_ms': det_result['time_ms'],
            'total_time_ms': enh_result['metrics']['time_ms'] + det_result['time_ms'],
            'metrics': enh_result['metrics']
        }

        if include_images:
            result['annotated_base64'] = _image_to_base64(annotated)
            result['enhanced_base64'] = _image_to_base64(enhanced_img)

        return result
    
    def _compute_brightness(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))
    
    def _compute_contrast(self, image: np.ndarray) -> float:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))
    
    def _init_aggregated_stats(self, enhancer_names: List[str]) -> Dict[str, Any]:
        stats = {
            'original': {
                'total_count': 0,
                'total_confidence': 0.0,
                'total_detection_time': 0.0,
                'brightness_sum': 0.0,
                'contrast_sum': 0.0
            },
            'enhanced': {}
        }
        for name in enhancer_names:
            stats['enhanced'][name] = {
                'total_count': 0,
                'total_confidence': 0.0,
                'total_enhancement_time': 0.0,
                'total_detection_time': 0.0,
                'total_time': 0.0,
                'brightness_sum': 0.0,
                'contrast_sum': 0.0
            }
        return stats
    
    def _update_aggregated_stats(self, stats: Dict, result: Dict):
        # Original
        orig = result['original']
        stats['original']['total_count'] += orig['count']
        stats['original']['total_confidence'] += orig['avg_confidence']
        stats['original']['total_detection_time'] += orig['detection_time_ms']
        stats['original']['brightness_sum'] += orig['metrics']['brightness']
        stats['original']['contrast_sum'] += orig['metrics']['contrast']
        
        # Enhanced
        for name, data in result['enhanced'].items():
            if name in stats['enhanced']:
                s = stats['enhanced'][name]
                s['total_count'] += data['count']
                s['total_confidence'] += data['avg_confidence']
                s['total_enhancement_time'] += data['enhancement_time_ms']
                s['total_detection_time'] += data['detection_time_ms']
                s['total_time'] += data['total_time_ms']
                s['brightness_sum'] += data['metrics']['brightness_after']
                s['contrast_sum'] += data['metrics']['contrast_after']
    
    def _finalize_aggregated_stats(self, stats: Dict, num_images: int) -> Dict:
        """Превращает суммы в средние значения и добавляет производные метрики."""
        if num_images == 0:
            return stats
        
        # Original averages
        o = stats['original']
        o['avg_count'] = o['total_count'] / num_images
        o['avg_confidence'] = o['total_confidence'] / num_images
        o['avg_detection_time_ms'] = o['total_detection_time'] / num_images
        o['avg_brightness'] = o['brightness_sum'] / num_images
        o['avg_contrast'] = o['contrast_sum'] / num_images
        
        # Enhanced averages
        for name, s in stats['enhanced'].items():
            s['avg_count'] = s['total_count'] / num_images
            s['avg_confidence'] = s['total_confidence'] / num_images
            s['avg_enhancement_time_ms'] = s['total_enhancement_time'] / num_images
            s['avg_detection_time_ms'] = s['total_detection_time'] / num_images
            s['avg_total_time_ms'] = s['total_time'] / num_images
            s['avg_brightness'] = s['brightness_sum'] / num_images
            s['avg_contrast'] = s['contrast_sum'] / num_images
            
            # Прирост относительно оригинала
            s['count_gain'] = s['avg_count'] - o['avg_count']
            s['count_gain_percent'] = (s['count_gain'] / o['avg_count'] * 100) if o['avg_count'] > 0 else 0
            s['confidence_gain'] = s['avg_confidence'] - o['avg_confidence']
            s['brightness_gain'] = s['avg_brightness'] - o['avg_brightness']
            s['contrast_gain'] = s['avg_contrast'] - o['avg_contrast']
        
        return stats