import cv2
import numpy as np  # Import numpy for type hinting

from . import read_img, save_img

class ContrastAdjustment:
    """
    Class for adjusting the contrast of an image.
    
    Attributes:
        image (numpy.ndarray): Input image.
        final_image (numpy.ndarray): Output image.
    """
    
    def __init__(self) -> None:
        """
        Initializes the ContrastAdjustment class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def adjust_contrast(self, alpha: float = 5.0, beta: int = 5) -> np.ndarray:
        """
        Adjusts the image's contrast and brightness.
        
        Args:
            alpha (float): Contrast factor (
                                            1.0 for no change, 
                                            < 1.0 decreases, 
                                            > 1.0 increases contrast).
            beta (int): Brightness factor (adjusts overall brightness).

        Returns:
            numpy.ndarray: adjusted contrast image.
        """
        self.final_image = cv2.convertScaleAbs(self.image, alpha=alpha, beta=beta)
        return self.final_image
        
    def save(self, filename: str) -> None:
        """
        Saves the adjusted image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = ContrastAdjustment()
    final_image = original_image.adjust_contrast()
    original_image.save('ContrastAdjustment')

    # Display original and adjusted images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()