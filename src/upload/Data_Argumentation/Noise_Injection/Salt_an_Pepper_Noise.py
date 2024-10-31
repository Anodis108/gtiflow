import cv2 
import numpy as np

from src.upload.common.read_save_img import read_img, save_img

class SaltAndPepperNoise():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def add_noise(self, salt_prob = 0.02, pepper_prob = 0.02):
        """
        Thêm nhiễu muối và tiêu vào ảnh.
        
        :param salt_prob: Xác suất cho các điểm nhiễu muối (giá trị 1).
        :param pepper_prob: Xác suất cho các điểm nhiễu tiêu (giá trị 0).
        """
        # Tạo bản sao của ảnh gốc để thêm nhiễu
        noisy_image = np.copy(self.image)

        # Tính số lượng pixel nhiễu muối và tiêu
        total_pixels = noisy_image.size
        num_salt = int(total_pixels * salt_prob)
        num_pepper = int(total_pixels * pepper_prob)

        # Thêm nhiễu muối
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1], :] = 1  # Giá trị muối là 1 (trắng)

        # Thêm nhiễu tiêu
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1], :] = 0  # Giá trị tiêu là 0 (đen)
        
        self.final_image = noisy_image
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = SaltAndPepperNoise()

    final_image = original_image.add_noise()
    original_image.save('SaltAndPepperNoise')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()