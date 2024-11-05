import cv2
import numpy as np

from . import read_img, save_img

class Cropping:
    """
    Class for cropping a specified region from an image.
    
    Attributes:
        image (numpy.ndarray): Input image .
        final_image (numpy.ndarray): Output image.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Cropping class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None  

    def crop(self, x: int = 0, y: int = 0, width: int = 10, height: int = 10) -> np.ndarray:
        """
        Crops a specified region of the image from (x, y) with given width and height.
        
        Args:
            x (int): The x-coordinate of the top-left corner for cropping. Default is 0.
            y (int): The y-coordinate of the top-left corner for cropping. Default is 0.
            width (int): The width of the cropped region. Default is 10.
            height (int): The height of the cropped region. Default is 10.
        
        Returns:
            numpy.ndarray: cropped section image.
        """
        self.final_image = self.image[y:y + height, x:x + width]
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the cropped image with the specified filename.
        
        Args:
            filename (str): Name of the output file, prefixed with 'violet'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    cropper = Cropping()

    cropped_image = cropper.crop()
    cropper.save('Cropping')

    # Display original and cropped images
    cv2.imshow('Original Image', cropper.image)
    cv2.imshow('After Image', cropped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()