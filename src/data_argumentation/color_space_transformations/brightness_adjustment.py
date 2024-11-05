import cv2
import numpy as np 

from common import read_img, save_img

class BrightnessAdjustment:
    """
    Class for adjusting the brightness of an image.
    
    Attributes:
        image (numpy.ndarray): The input image 
        final_image (numpy.ndarray): The output image
    """
    
    def __init__(self) -> None:
        """
        Initializes the BrightnessAdjustment class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None  

    def adjust_brightness(self, beta: int = 10) -> np.ndarray:
        """
        Adjusts the brightness of the image based on the beta value.

        Args:
            beta (int): Brightness adjustment value (default = 10).
                        Positive values increase brightness, 
                        negative values decrease it.

        Returns:
            numpy.ndarray: The brightness-adjusted image.
        
        Note:
            Brightness is modified by adding 'beta' to all pixels. Alpha is fixed at 1, so contrast remains unchanged.
        """
        adjusted_image: np.ndarray = cv2.convertScaleAbs(self.image, alpha=1, beta=beta)
        self.final_image = adjusted_image
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the brightness-adjusted image 

        Args:
            filename (str): The output filename for saving the image, prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    # Initialize the class and adjust the image brightness
    original_image = BrightnessAdjustment()
    final_image = original_image.adjust_brightness()
    original_image.save('BrightnessAdjustment')

    # Display the original and adjusted images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()