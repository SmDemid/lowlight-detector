import cv2
import numpy as np
from .base import BaseEnhancer
from .clahe import CLAHEEnhancer

class BilateralEnhancer(BaseEnhancer):
    def __init__(self, d: int = 9, sigma_color: float = 75, sigma_space: float = 75):
        super().__init__("Bilateral + CLAHE")
        self.d = d
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.clahe = CLAHEEnhancer(clip_limit=1.5)  # уменьшенный clip limit
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        # Билатеральная фильтрация для шумоподавления
        filtered = cv2.bilateralFilter(image, self.d, self.sigma_color, self.sigma_space)
        # Затем применяем CLAHE для повышения контраста
        enhanced = self.clahe.enhance(filtered)
        return enhanced