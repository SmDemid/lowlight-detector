import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from .base import BaseEnhancer

# Определение архитектуры Zero-DCE (упрощённая версия из оригинального репозитория)
class DCE_Net(nn.Module):
    def __init__(self, n_filters=32):
        super(DCE_Net, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(3, n_filters, 3, padding=1)
        self.conv2 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv3 = nn.Conv2d(n_filters, n_filters, 3, padding=1)
        self.conv4 = nn.Conv2d(n_filters*2, n_filters, 3, padding=1)
        self.conv5 = nn.Conv2d(n_filters*2, n_filters, 3, padding=1)
        self.conv6 = nn.Conv2d(n_filters*2, 3, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.maxpool(x1)
        x2 = self.relu(self.conv2(x2))
        x3 = self.maxpool(x2)
        x3 = self.relu(self.conv3(x3))
        x3_up = self.upsample(x3)
        x4 = torch.cat([x2, x3_up], dim=1)
        x4 = self.relu(self.conv4(x4))
        x4_up = self.upsample(x4)
        x5 = torch.cat([x1, x4_up], dim=1)
        x5 = self.relu(self.conv5(x5))
        x6 = self.conv6(x5)
        return torch.tanh(x6)

class ZeroDCEEnhancer(BaseEnhancer):
    def __init__(self, model_path: str = 'models/zero_dce.pth', iterations: int = 8):
        super().__init__("Zero-DCE")
        self.model_path = model_path
        self.iterations = iterations
        self.device = torch.device('cpu')
        self.model = None
        self._load_model()
    
    def _load_model(self):
        self.model = DCE_Net().to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            print(f"Warning: Zero-DCE weights not found at {self.model_path}. Using random weights.")
        self.model.eval()
    
    def enhance(self, image: np.ndarray) -> np.ndarray:
        # Преобразование в тензор
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img_rgb).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Получаем кривые усиления
            curve = self.model(img_tensor)
            # Применяем итеративно
            enhanced = img_tensor
            for _ in range(self.iterations):
                enhanced = enhanced + curve * (1 - enhanced)  # формула из Zero-DCE
            enhanced = torch.clamp(enhanced, 0, 1)
        
        # Обратно в numpy
        enhanced_np = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = (enhanced_np * 255).astype(np.uint8)
        return cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)