import cv2 

from src.upload.common.read_save_img import read_img, save_img

class ContrastAdjustment():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def adjust_contrast(self,alpha=5, beta=5):
        """
        Điều chỉnh độ tương phản của ảnh.
        alpha: Hệ số điều chỉnh độ tương phản (1.0 là không thay đổi, < 1.0 làm giảm độ tương phản, > 1.0 làm tăng độ tương phản)
        beta: Hệ số điều chỉnh độ sáng (thay đổi độ sáng của ảnh)
        """
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        
        self.final_image = adjusted_image
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = ContrastAdjustment()

    final_image = original_image.adjust_contrast()
    original_image.save('ContrastAdjustment')

    # Hiển thị ảnh gốc và ảnh đã xử lý

    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

