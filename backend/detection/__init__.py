from .yolo_detector import YOLODetector
from .rcnn_detector import FasterRCNNDetector

def get_detector(name: str, confidence_threshold: float = 0.25):
    if name.startswith('yolov8'):
        size = name[-1]  # 'n' или 'm'
        return YOLODetector(model_size=size, confidence_threshold=confidence_threshold)
    elif name == 'faster_rcnn':
        return FasterRCNNDetector(confidence_threshold=confidence_threshold)
    else:
        raise ValueError(f"Unknown detector: {name}")