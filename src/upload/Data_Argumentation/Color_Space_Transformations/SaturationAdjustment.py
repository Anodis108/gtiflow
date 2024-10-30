import cv2 
import numpy as np
from src.upload.common.read_save_img import read_img, save_img

class SaturationAdjustment():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def adjust_saturation(self, saturation_scale=1.2):
        """
        Điều chỉnh độ bão hòa của ảnh.
        saturation_scale: 
            > 1.0: Tăng độ bão hòa
            = 1.0: Không thay đổi
            < 1.0: Giảm độ bão hòa
        """
        # Chuyển ảnh từ không gian màu BGR sang HSV
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Tăng hoặc giảm độ bão hòa
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255) #  lấy dữ liệu của kênh độ bão hòa (kênh 1 - kênh thứ hai) trong ảnh HSV
        # clip: giới hạn trong khoảng 0 - 255
        
        # Chuyển ảnh từ HSV về BGR
        adjusted_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        
        self.final_image = adjusted_image
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = SaturationAdjustment()

    final_image = original_image.adjust_saturation()
    original_image.save('SaturationAdjustment')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.src)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()