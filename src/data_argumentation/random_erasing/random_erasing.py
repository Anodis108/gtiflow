import cv2
import numpy as np
import random

from . import read_img, save_img

class RandomErasing:
    """
    Class for applying random erasing data augmentation to an image.
    
    Attributes:
        image (numpy.ndarray): Input image.
        scale (tuple): Scale range for erased area.
        ratio (tuple): Aspect ratio range for erased area.
        final_image (numpy.ndarray): Image after applying random erasing.
    """
    
    def __init__(self, scale: tuple[float, float] = (0.02, 0.33), ratio: tuple[float, float] = (0.3, 3.3)) -> None:
        """
        Initializes the RandomErasing class and loads the input image.
        
        Args:
            scale (tuple): Range for the size of the erased area.
            ratio (tuple): Range for the aspect ratio of the erased area.
        """
        self.image: np.ndarray = read_img()  # Read image using read_img
        self.scale: tuple[float, float] = scale  # Size scale for erasure
        self.ratio: tuple[float, float] = ratio  # Aspect ratio for erasure
        self.final_image: np.ndarray = self.image.copy()  # Copy original image for processing

    def apply(self) -> np.ndarray:
        """
        Applies random erasing to the image.
        
        Returns:
            numpy.ndarray: Image after random erasing.
        """
        h, w, _ = self.image.shape
        area = h * w

        # Choose a random scale
        target_area = random.uniform(self.scale[0], self.scale[1]) * area
        aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate height and width of the erased area
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        # Ensure height and width do not exceed image dimensions
        if h_erase > h or w_erase > w:
            return self.image  # Return original image if erasure is not possible

        # Choose a random position to erase
        x1 = random.randint(0, w - w_erase)
        y1 = random.randint(0, h - h_erase)

        # Create a random erase color
        erase_color = np.random.randint(0, 256, (h_erase, w_erase, 3), dtype=np.uint8)
        self.final_image[y1:y1 + h_erase, x1:x1 + w_erase] = erase_color

        return self.final_image
        
    def save(self, filename: str) -> None:
        """
        Saves the processed image with random erasing.
        
        Args:
            filename (str): The output filename prefixed with 'erased_'.
        """
        save_img(save_img_name='erased_' + filename, img=self.final_image)

if __name__ == "__main__":
    # Initialize and apply Random Erasing
    random_erasing = RandomErasing()
    final_image = random_erasing.apply()

    # Save the processed image
    random_erasing.save('RandomErasing')

    # Display original and processed images
    cv2.imshow('Original Image', random_erasing.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()