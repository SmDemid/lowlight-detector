import cv2
import numpy as np
from .base import BaseEnhancer

class MSRCREnhancer(BaseEnhancer):
    def __init__(self, scales: list = [15, 80, 250], alpha: float = 125.0, beta: float = 46.0):
        super().__init__("MSRCR")
        self.scales = scales
        self.alpha = alpha
        self.beta = beta
    
    def _single_scale_retinex(self, img: np.ndarray, sigma: float) -> np.ndarray:
        """Одномасштабный Retinex."""
        blur = cv2.GaussianBlur(img, (0, 0), sigma)
        retinex = np.log10(img + 1.0) - np.log10(blur + 1.0)
        return retinex
    
    def _color_restoration(self, img: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Восстановление цвета."""
        img_sum = np.sum(img, axis=2, keepdims=True) + 1.0
        color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum))
        return color_restoration
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float32) + 1.0
        
        # Multi-Scale Retinex
        msr = np.zeros_like(img_float)
        for sigma in self.scales:
            msr += self._single_scale_retinex(img_float, sigma)
        msr /= len(self.scales)
        
        # Color Restoration
        cr = self._color_restoration(img_float, self.alpha, self.beta)
        msrcr = msr * cr
        
        # Постобработка: нормализация в диапазон [0, 255]
        for i in range(3):
            channel = msrcr[:, :, i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:
                msrcr[:, :, i] = (channel - min_val) / (max_val - min_val) * 255.0
            else:
                msrcr[:, :, i] = 0
        
        result = np.clip(msrcr, 0, 255).astype(np.uint8)
        return result