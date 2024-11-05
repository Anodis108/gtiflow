import cv2 
import numpy as np
import random

from common import read_img, save_img, IMAGE_PATH2

class ImageMixer:
    """
    Class for mixing two images based on a specified weight.
    
    Attributes:
        image1 (numpy.ndarray): First input image.
        image2 (numpy.ndarray): Second input image.
        final_image (numpy.ndarray): Resulting mixed image.
    """

    def __init__(self) -> None:
        """
        Initializes the ImageMixer class and loads the input images.
        
        """
        self.image1: np.ndarray = read_img()  
        self.image2: np.ndarray = read_img(img_path=IMAGE_PATH2)  
        self.final_image: np.ndarray = None 

    def mix_images(self, alpha: float = 0.5) -> np.ndarray:
        """
        Mixes two images with a specified weight.
        
        Args:
            alpha (float): Weight for the first image (0 <= alpha <= 1).
        
        Returns:
            numpy.ndarray: The mixed image.
        
        Raises:
            ValueError: If the two images do not have the same dimensions.
        """
        if self.image1.shape != self.image2.shape:
            raise ValueError("Both images must have the same size to mix.")
        
        # Mix images
        self.final_image = cv2.addWeighted(self.image1, alpha, self.image2, 1 - alpha, 0)
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the mixed image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = ImageMixer()

    final_image = original_image.mix_images()
    original_image.save('ImageMixer')

    # Display original and mixed images
    cv2.imshow('Original Image', original_image.image1)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()