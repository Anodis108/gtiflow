import cv2 
from src.upload.common.read_save_img import read_img, save_img

class Flipping():
    def __init__(self) -> None:
        self.src = read_img()
        self.flipped_image = None

        
    def flip(self, flip_code=1):
        """
        Lật ảnh theo hướng chỉ định.
        flip_code:
            0  - Lật theo trục x (dọc)
            1  - Lật theo trục y (ngang)
           -1  - Lật theo cả hai trục
        """
        self.flipped_image = cv2.flip(self.image, flip_code)
        return self.flipped_image
        
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.rotated_img)
        
if __name__ == "__main__":
    flipper = Flipping()

    # lật ảnh 
    flipped_image = flipper.flip()
    # Lưu ảnh đã lật 
    flipper.save('Flipping')

    
    # Hiển thị ảnh gốc và ảnh đã xử lý
    cv2.imshow('Original Image', flipper.src)
    cv2.imshow('After Image', flipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()