import cv2
import numpy as np

from . import read_img, save_img

class SaturationAdjustment:
    """
    Class for adjusting the saturation of an image.
    
    Attributes:
        image (numpy.ndarray): Input image.
        final_image (numpy.ndarray): Output image.
    """

    def __init__(self) -> None:
        """
        Initializes the SaturationAdjustment class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def adjust_saturation(self, saturation_scale: float = 1.2) -> np.ndarray:
        """
        Adjusts the saturation of the image.
        
        Args:
            saturation_scale (float): Factor to adjust saturation.
                                      > 1.0 increases saturation.
                                      = 1.0 maintains original saturation.
                                      < 1.0 decreases saturation.
        
        Returns:
            numpy.ndarray: adjusted saturation image.
        """
        # Convert image from BGR to HSV color space
        hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        
        # Modify saturation channel
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * saturation_scale, 0, 255)
        
        # Convert image back to BGR color space
        self.final_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
        return self.final_image
        
    def save(self, filename: str) -> None:
        """
        Saves the adjusted image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = SaturationAdjustment()
    final_image = original_image.adjust_saturation()
    original_image.save('SaturationAdjustment')

    # Display original and adjusted images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()