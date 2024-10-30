import cv2 
import numpy as np
import random
from src.upload.common.read_save_img import read_img, save_img

class ColorJitter():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def brightness(self, factor):
        # Thay đổi độ sáng
        return cv2.convertScaleAbs(self.image, alpha=1, beta=factor)

    def contrast(self, factor):
        # Thay đổi độ tương phản
        return cv2.convertScaleAbs(self.image, alpha=factor, beta=0)

    def saturation(self, factor):
        # Thay đổi độ bão hòa
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)  # Thay đổi kênh bão hòa
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def jitter(self, brightness_factor=0, contrast_factor=1, saturation_factor=1):
        # Áp dụng các biến đổi màu sắc random
        img = self.brightness(brightness_factor)
        img = self.contrast(contrast_factor)
        img = self.saturation(saturation_factor)
        
        self.final_image= img
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = ColorJitter()

    # Thay đổi các thông số
    brightness_factor = random.randint(-50, 50)  # Độ sáng ngẫu nhiên
    contrast_factor = random.uniform(0.5, 1.5)   # Độ tương phản ngẫu nhiên
    saturation_factor = random.uniform(0.5, 1.5)  # Độ bão hòa ngẫu nhiên

    # Thực hiện biến đổi màu sắc
    jittered_image = original_image.jitter(brightness_factor, contrast_factor, saturation_factor)
    original_image.save('ColorJitter')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.src)
    cv2.imshow('After Image', jittered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() # note