import cv2 
import numpy as np

from src.upload.common.read_save_img import read_img, save_img

class Translation():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None


        
    def translate(self, x=10, y=10):
        """
        Dịch chuyển ảnh.
        :param x: Khoảng cách dịch chuyển theo trục x (có thể dương hoặc âm -> phải hoặc trái).
        :param y: Khoảng cách dịch chuyển theo trục y (có thể dương hoặc âm -> xuống hoặc lên).
        """
        # Tạo ma trận dịch chuyển
        translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
        # Áp dụng dịch chuyển lên ảnh
        self.final_image = cv2.warpAffine(self.image, translation_matrix, (self.image.shape[1], self.image.shape[0]))
        
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = Translation()

    final_image = original_image.translate()
    original_image.save('Translation')

    # Hiển thị ảnh gốc và ảnh đã xử lý

    cv2.imshow('Original Image', original_image.image)

    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()