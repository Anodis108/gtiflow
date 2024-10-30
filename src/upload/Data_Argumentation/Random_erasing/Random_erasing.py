import cv2
import numpy as np
import random

from src.upload.common.read_save_img import read_img, save_img

class RandomErasing:
    def __init__(self, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        # Đọc ảnh từ hàm read_img
        self.image = read_img()
        self.scale = scale  # Tỉ lệ kích thước vùng bị xóa
        self.ratio = ratio  # Tỉ lệ chiều dài/chiều rộng của vùng bị xóa
        self.final_image = self.image.copy()  # Sao chép ảnh gốc để xử lý

    def apply(self):
        h, w, _ = self.image.shape
        area = h * w

        # Chọn tỉ lệ ngẫu nhiên
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Tính chiều dài và chiều rộng của vùng bị xóa
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        # Đảm bảo chiều dài và chiều rộng không vượt quá kích thước ảnh
        if h_erase > h or w_erase > w:
            return self.image  # Trả về ảnh gốc nếu không thể xóa

        # Chọn vị trí ngẫu nhiên để xóa
        x1 = random.randint(0, w - w_erase)
        y1 = random.randint(0, h - h_erase)

        # Tạo một vùng màu ngẫu nhiên
        erase_color = np.random.randint(0, 256, (h_erase, w_erase, 3), dtype=np.uint8)
        self.final_image[y1:y1 + h_erase, x1:x1 + w_erase] = erase_color

        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='erased_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    # Khởi tạo và áp dụng Random Erasing
    random_erasing = RandomErasing()
    final_image = random_erasing.apply()

    # Lưu ảnh đã qua xử lý
    random_erasing.save('RandomErasing')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', random_erasing.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
