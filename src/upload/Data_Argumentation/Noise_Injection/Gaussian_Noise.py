import cv2 
import numpy as np

from src.upload.common.read_save_img import read_img, save_img

class GaussianNoise():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def add_gaussian_noise(self, mean=0, sigma=25):
        """
        Thêm Gaussian Noise vào ảnh.
        mean: Giá trị trung bình của noise
        sigma: Độ lệch chuẩn của noise
        """
        # Tạo Gaussian noise
        gauss = np.random.normal(mean, sigma, self.image.shape).astype(np.uint8)
        
        # Thêm noise vào ảnh
        self.final_image = cv2.add(self.image, gauss)
        
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = GaussianNoise()

    final_image = original_image.add_gaussian_noise()
    original_image.save('GaussianNoise')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()