import cv2
import numpy as np
from .base import BaseEnhancer

class CLAHEEnhancer(BaseEnhancer):
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)):
        super().__init__("CLAHE")
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, 
                                 tileGridSize=self.tile_grid_size)
        l_eq = clahe.apply(l)
        
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)