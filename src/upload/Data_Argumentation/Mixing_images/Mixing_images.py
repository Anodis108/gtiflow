import cv2 
import numpy as np
import random

from src.upload.common.read_save_img import read_img, save_img
from src.upload.common.constants import IMAGE_PATH2

class ImageMixer():
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.image1 = read_img()
        self.image2 = read_img(img_path=IMAGE_PATH2)
        
        self.final_image = read_img

    def mix_images(self, alpha=0.5):
        """
        Trộn hai ảnh với trọng số alpha.
        alpha: Trọng số cho ảnh thứ nhất (0 <= alpha <= 1).
                Trọng số cho ảnh thứ hai sẽ là (1 - alpha).
        """
        if self.image1.shape != self.image2.shape:
            raise ValueError("Hai ảnh phải có cùng kích thước để trộn.")
        
        # Trộn ảnh
        self.final_image = cv2.addWeighted(self.image1, alpha, self.image2, 1 - alpha, 0)


        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = ImageMixer()

    final_image = original_image.mix_images()
    original_image.save('ImageMixer')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.src)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()