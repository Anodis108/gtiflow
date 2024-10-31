import cv2 

from src.upload.common.read_save_img import read_img, save_img

class BrightnessAdjustment():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def adjust_brightness(self,beta = 10):
        """
        Điều chỉnh độ sáng của ảnh.
        
        :param beta: Giá trị điều chỉnh độ sáng. 
                     Giá trị dương sẽ làm tăng độ sáng, 
                     giá trị âm sẽ làm giảm độ sáng.
        """
        # Thay đổi độ sáng bằng cách cộng beta vào tất cả các pixel, alpha là độ tương phản nên không liên quan
        adjusted_image = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)
        
        self.final_image = adjusted_image
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.final_image)
        
if __name__ == "__main__":
    original_image  = BrightnessAdjustment()

    final_image = original_image.adjust_brightness()
    original_image.save('BrightnessAdjustment')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows() # note