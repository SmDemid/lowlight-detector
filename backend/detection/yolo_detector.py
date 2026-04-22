from ultralytics import YOLO
import numpy as np
import os
from .base import BaseDetector

class YOLODetector(BaseDetector):
    def __init__(self, model_size: str = 'm', confidence_threshold: float = 0.25):
        super().__init__(f"YOLOv8{model_size}", confidence_threshold)
        
        # Определяем путь к папке models (на два уровня выше)
        # backend/detection/ -> backend/ -> корень проекта
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        model_filename = f'yolov8{model_size}.pt'
        model_path = os.path.join(models_dir, model_filename)
        
        # Проверяем, есть ли модель в папке models
        if os.path.exists(model_path):
            print(f"[INFO] Loading YOLO from: {model_path}")
            self.model = YOLO(model_path)
        else:
            # Если нет, загружаем и сохраняем в models
            print(f"[INFO] Downloading YOLOv8{model_size} to: {model_path}")
            self.model = YOLO(model_filename)
            # Перемещаем скачанный файл в папку models (опционально)
            downloaded_path = os.path.join(os.getcwd(), model_filename)
            if os.path.exists(downloaded_path) and downloaded_path != model_path:
                import shutil
                shutil.move(downloaded_path, model_path)
                print(f"[INFO] Moved model to: {model_path}")
        
        # Принудительно используем CPU (если нужно)
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