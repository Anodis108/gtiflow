import cv2 
from src.upload.common.read_save_img import read_img, save_img

class Cropping():
    def __init__(self) -> None:
        self.image = read_img()
        self.final_image = None

        
    def crop(self, x=0, y=0, width=10, height=10):
        """
        Cắt ảnh từ tọa độ (x, y) với kích thước width x height.
        """
        cropped_image = self.image[y:y + height, x:x + width]
        self.final_image = cropped_image
        
        return self.final_image
        
    def save(self, filename):
        save_img(save_img_name='violet' + filename, img=self.final_image)
        
if __name__ == "__main__":
    cropper  = Cropping()

    cropped_image = cropper.crop()
    cropper.save('Cropping')

    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', cropper.src)
    cv2.imshow('After Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()