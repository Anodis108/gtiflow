import cv2 
import numpy as np

from common import read_img, save_img

class Translation:
    """
    Class for translating (shifting) an image.

    Attributes:
        image (numpy.ndarray): Input image.
        final_image (numpy.ndarray): Output image.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Translation class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def translate(self, x: int = 10, y: int = 10) -> np.ndarray:
        """
        Translates the image by specified distances along x and y axes.
        
        Args:
            x (int): Distance to translate along the x-axis (positive for right, negative for left).
            y (int): Distance to translate along the y-axis (positive for down, negative for up).
        
        Returns:
            numpy.ndarray: Translated image.
        """
        translation_matrix = np.float32([[1, 0, x], [0, 1, y]])
        self.final_image = cv2.warpAffine(self.image, translation_matrix, (self.image.shape[1], self.image.shape[0]))
        
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the translated image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = Translation()

    final_image = original_image.translate()
    original_image.save('Translation')

    # Display original and translated images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroy