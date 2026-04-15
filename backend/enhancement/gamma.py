import cv2
import numpy as np
from .base import BaseEnhancer

class GammaEnhancer(BaseEnhancer):
    def __init__(self, target_brightness: float = 128.0):
        super().__init__("Gamma Correction")
        self.target_brightness = target_brightness
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        # Автоматический подбор гаммы
        if mean_brightness > 0:
            gamma = np.log(self.target_brightness / 255.0) / np.log(mean_brightness / 255.0)
            gamma = np.clip(gamma, 0.2, 3.0)
        else:
            gamma = 1.0
        
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype(np.uint8)
        return cv2.LUT(image, table)