from ultralytics import YOLO
import numpy as np
from .base import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_size: str = 'm', confidence_threshold: float = 0.25):
        super().__init__(f"YOLOv8{model_size}", confidence_threshold)
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.model.to('cpu')
    
    def detect(self, image: np.ndarray) -> list:
        results = self.model(image, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names[class_id]
                    
                    detections.append({
                        'class': class_name,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2]
                    })
        return detections