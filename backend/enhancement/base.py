from abc import ABC, abstractmethod
import cv2
import time
import numpy as np
from typing import Dict, Any

class BaseEnhancer(ABC):
    """Базовый класс для всех методов улучшения изображений."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def enhance(self, image: np.ndarray) -> np.ndarray:
        """Применяет улучшение к изображению. Должен быть реализован в подклассе."""
        pass
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """Полный цикл обработки с замером времени и вычислением метрик."""
        start_time = time.time()
        enhanced = self.enhance(image)
        elapsed_ms = (time.time() - start_time) * 1000
        
        metrics = self._compute_metrics(image, enhanced)
        metrics['time_ms'] = round(elapsed_ms, 2)
        
        return {
            'image': enhanced,
            'metrics': metrics,
            'method_name': self.name
        }
    
    def _compute_metrics(self, original: np.ndarray, enhanced: np.ndarray) -> Dict[str, Any]:
        """Вычисляет метрики качества: средняя яркость, контраст (std), PSNR."""
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        
        orig_mean = np.mean(orig_gray)
        enh_mean = np.mean(enh_gray)
        orig_std = np.std(orig_gray)
        enh_std = np.std(enh_gray)
        
        # PSNR
        mse = np.mean((orig_gray.astype(float) - enh_gray.astype(float)) ** 2)
        if mse > 0:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        return {
            'brightness_before': round(orig_mean, 2),
            'brightness_after': round(enh_mean, 2),
            'brightness_gain': round(enh_mean - orig_mean, 2),
            'contrast_before': round(orig_std, 2),
            'contrast_after': round(enh_std, 2),
            'contrast_gain': round(enh_std - orig_std, 2),
            'psnr': round(psnr, 2) if psnr != float('inf') else None
        }