from abc import ABC, abstractmethod
import cv2
import time
import numpy as np
from typing import List, Dict, Any

class BaseDetector(ABC):
    def __init__(self, name: str, confidence_threshold: float = 0.25):
        self.name = name
        self.confidence_threshold = confidence_threshold
    
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Возвращает список словарей с информацией о детекциях:
        [
            {
                'class': str,
                'confidence': float,
                'bbox': [x1, y1, x2, y2]  # координаты в пикселях
            },
            ...
        ]
        """
        pass
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        start_time = time.time()
        detections = self.detect(image)
        elapsed_ms = (time.time() - start_time) * 1000
        
        annotated = self._draw_detections(image.copy(), detections)
        
        avg_conf = np.mean([d['confidence'] for d in detections]) if detections else 0.0
        
        return {
            'detections': detections,
            'count': len(detections),
            'avg_confidence': round(avg_conf, 3),
            'time_ms': round(elapsed_ms, 2),
            'annotated_image': annotated
        }
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Цвет по имени класса (простой хэш)
            class_id = hash(det['class']) % 255
            color = (class_id, 255 - class_id, (class_id * 2) % 255)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image