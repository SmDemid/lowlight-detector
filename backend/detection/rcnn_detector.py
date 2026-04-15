import torch
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights
import numpy as np
import cv2
from .base import BaseDetector

class FasterRCNNDetector(BaseDetector):
    def __init__(self, confidence_threshold: float = 0.25):
        super().__init__("Faster R-CNN", confidence_threshold)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).to(self.device)
        self.model.eval()
        self.categories = weights.meta["categories"]
    
    def detect(self, image: np.ndarray) -> list:
        # Преобразование в тензор
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)
        
        detections = []
        for box, score, label in zip(predictions[0]['boxes'], predictions[0]['scores'], predictions[0]['labels']):
            if score >= self.confidence_threshold:
                x1, y1, x2, y2 = box.cpu().numpy().astype(int).tolist()
                class_name = self.categories[label.item()]
                detections.append({
                    'class': class_name,
                    'confidence': float(score.cpu().numpy()),
                    'bbox': [x1, y1, x2, y2]
                })
        return detections