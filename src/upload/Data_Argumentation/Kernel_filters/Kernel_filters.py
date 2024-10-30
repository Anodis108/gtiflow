import cv2 

from src.upload.common.read_save_img import read_img, save_img

class KernelFilter():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def apply_gaussian_filter(self, kernel_size=(5, 5), sigma_x=0):
        """
        Áp dụng bộ lọc Gaussian lên ảnh.
        :param kernel_size: Kích thước của kernel (phải là số lẻ).
        :param sigma_x: Độ lệch chuẩn theo trục x.
        :return: Ảnh đã lọc.
        """
        self.final_image =  cv2.GaussianBlur(self.image, kernel_size, sigma_x)
        return self.final_image

    def apply_median_filter(self, kernel_size=5):
        """
        Áp dụng bộ lọc Median lên ảnh.
        :param kernel_size: Kích thước của kernel (phải là số lẻ).
        :return: Ảnh đã lọc.
        """
        self.final_image =  cv2.medianBlur(self.image, kernel_size)
        return self.final_image

    def apply_bilateral_filter(self, d=9, sigma_color=75, sigma_space=75):
        """
        Áp dụng bộ lọc Bilateral lên ảnh.
        :param d: Kích thước của kernel.
        :param sigma_color: Độ lệch chuẩn cho màu sắc.
        :param sigma_space: Độ lệch chuẩn cho không gian.
        :return: Ảnh đã lọc.
        """
        self.final_image =  cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = KernelFilter()

    final_image = original_image.apply_bilateral_filter()
    final_image = original_image.apply_gaussian_filter()
    final_image = original_image.apply_median_filter()
    original_image.save('KernelFilter')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.src)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()