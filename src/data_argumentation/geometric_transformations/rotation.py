import cv2
import numpy as np

from . import read_img, save_img

class Rotation:
    """
    Class for rotating an image by a specified angle.
    
    Attributes:
        src (numpy.ndarray): Input image.
        rotated_img (numpy.ndarray): Output image   .
    """
    
    def __init__(self) -> None:
        """
        Initializes the Rotation class and loads the input image.
        """
        self.src: np.ndarray = read_img()  
        self.rotated_img: np.ndarray = None  
        
    def rotate(self, angle: float = 45.0, scale: float = 1.0) -> np.ndarray:
        """
        Rotates the image by a specified angle and scale.
        
        Args:
            angle (float): The angle of rotation in degrees.
            scale (float): The scaling factor (default is 1.0).
        
        Returns:
            numpy.ndarray: The rotated image.
        """
        (h, w) = self.src.shape[:2]
        center = (w // 2, h // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Perform rotation
        self.rotated_img = cv2.warpAffine(self.src, rotation_matrix, (w, h))
        
        return self.rotated_img
        
    def save(self, filename: str) -> None:
        """
        Saves the rotated image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.rotated_img)

if __name__ == "__main__":
    rotator = Rotation()

    # Rotate the image
    rotated_image = rotator.rotate()
    # Save the rotated image
    rotator.save('rotated')

    # Display original and rotated images
    cv2.imshow('Original Image', rotator.src)
    cv2.imshow('Rotated Image', rotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()