import cv2
import numpy as np

from . import read_img, save_img

class Flipping:
    """
    Class for flipping an image in a specified direction.
    
    Attributes:
        image (numpy.ndarray): Input image.
        flipped_image (numpy.ndarray): Output image.
    """
    
    def __init__(self) -> None:
        """
        Initializes the Flipping class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.flipped_image: np.ndarray = None  

    def flip(self, flip_code: int = 1) -> np.ndarray:
        """
        Flips the image in the specified direction.

        Args:
            flip_code (int): Code to specify flip direction.
                0  - Flip around the x-axis (vertical flip).
                1  - Flip around the y-axis (horizontal flip).
               -1  - Flip around both axes.
        
        Returns:
            numpy.ndarray: Flipped image.
        """
        self.flipped_image = cv2.flip(self.image, flip_code)
        return self.flipped_image

    def save(self, filename: str) -> None:
        """
        Saves the flipped image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.flipped_image)

if __name__ == "__main__":
    flipper = Flipping()

    # Flip the image
    flipped_image = flipper.flip()
    # Save the flipped image
    flipper.save('Flipping')

    # Display original and flipped images
    cv2.imshow('Original Image', flipper.image)
    cv2.imshow('After Image', flipped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()