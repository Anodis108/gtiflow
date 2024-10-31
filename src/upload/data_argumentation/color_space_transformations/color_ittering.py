import cv2
import numpy as np
import random

from . import read_img, save_img

class ColorJitter:
    """
    Applies brightness, contrast, and saturation adjustments to an image.

    Attributes:
        image (numpy.ndarray): The input image 
        final_image (numpy.ndarray): The transformed image.
    """

    def __init__(self) -> None:
        """
        Initializes the ColorJitter class with an input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def brightness(self, factor: int) -> np.ndarray:
        """
        Adjusts image brightness.

        Args:
            factor (int): Amount to adjust brightness. 
                            Positive values increase brightness, 
                            negative values decrease it.

        Returns:
            numpy.ndarray: Image with adjusted brightness.
        """
        return cv2.convertScaleAbs(self.image, alpha=1, beta=factor)

    def contrast(self, factor: float) -> np.ndarray:
        """
        Adjusts image contrast.

        Args:
            factor (float): Contrast adjustment factor. 
                            Greater than 1 increases contrast, 
                            less than 1 decreases it.

        Returns:
            numpy.ndarray: Image with adjusted contrast.
        """
        return cv2.convertScaleAbs(self.image, alpha=factor, beta=0)

    def saturation(self, factor: float) -> np.ndarray:
        """
        Adjusts image saturation.

        Args:
            factor (float): Saturation adjustment factor. 
                            Greater than 1 increases saturation, 
                            less than 1 decreases it.

        Returns:
            numpy.ndarray: Image with adjusted saturation.
        """
        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def jitter(self, brightness_factor: int = 0, contrast_factor: float = 1, saturation_factor: float = 1) -> np.ndarray:
        """
        Applies color transformations with specified brightness, contrast, and saturation factors.

        Args:
            brightness_factor (int): Brightness adjustment factor.
            contrast_factor (float): Contrast adjustment factor.
            saturation_factor (float): Saturation adjustment factor.

        Returns:
            numpy.ndarray: Image after color jitter transformations.
        """
        img = self.brightness(brightness_factor)
        img = self.contrast(contrast_factor)
        img = self.saturation(saturation_factor)
        
        self.final_image = img
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the adjusted image to a file.

        Args:
            filename (str): The filename for saving the image, with prefix 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = ColorJitter()

    # Set random adjustment factors
    brightness_factor = random.randint(-50, 50)
    contrast_factor = random.uniform(0.5, 1.5)
    saturation_factor = random.uniform(0.5, 1.5)

    # Apply color jitter transformations
    jittered_image = original_image.jitter(brightness_factor, contrast_factor, saturation_factor)
    original_image.save('ColorJitter')

    # Display the original and adjusted images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', jittered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()