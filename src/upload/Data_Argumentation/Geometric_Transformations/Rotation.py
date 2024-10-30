import cv2 
from src.upload.common.read_save_img import read_img, save_img

class Rotation():
    def __init__(self) -> None:
        self.src = read_img()
        self.rotated_img = None
        
    def rotate(self, angle=45, scale=1.0):
        (h, w) = self.img.shape[:2]
        center = (w//2, h//2)
        
        # Tạo ma trận xoay
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Thực hiện xoay ảnh
        self.rotated_img = cv2.warpAffine(self.src, rotation_matrix, (w, h))
        
        return self.rotated_img
        
    def save(self, filename):
        save_img(save_img_name='violet_' + filename, img=self.rotated_img)
        
if __name__ == "__main__":
    rotator = Rotation()

    # Xoay ảnh 
    rotated_image = rotator.rotate()
    # Lưu ảnh đã xoay 
    rotator.save('rotated')

    
    # Hiển thị ảnh gốc và ảnh đã xoay
    cv2.imshow('Original Image', rotator.src)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()