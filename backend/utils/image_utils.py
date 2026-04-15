import cv2
import numpy as np
import base64

def image_to_base64(image: np.ndarray, ext='.jpg') -> str:
    """Конвертирует numpy-изображение в base64 строку для передачи в JSON."""
    _, buffer = cv2.imencode(ext, image)
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(b64_string: str) -> np.ndarray:
    """Декодирует base64 строку обратно в numpy-изображение."""
    img_bytes = base64.b64decode(b64_string)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def load_image_from_file(file_storage) -> np.ndarray:
    """Читает изображение из объекта FileStorage Flask."""
    np_arr = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Невозможно декодировать изображение")
    return img

def resize_image(image: np.ndarray, max_size: int = 1024) -> np.ndarray:
    """Изменяет размер изображения, сохраняя пропорции, если оно превышает max_size."""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image