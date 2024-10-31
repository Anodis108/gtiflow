import cv2 
import numpy as np

from . import read_img, save_img

class GaussianNoise:
    """
    Class for adding Gaussian noise to an image.
    
    Attributes:
        image (numpy.ndarray): Input image.
        final_image (numpy.ndarray): Image after adding noise.
    """
    
    def __init__(self) -> None:
        """
        Initializes the GaussianNoise class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def add_gaussian_noise(self, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
        """
        Adds Gaussian noise to the image.
        
        Args:
            mean (float): Mean value of the noise.
            sigma (float): Standard deviation of the noise.
        
        Returns:
            numpy.ndarray: Image with added Gaussian noise.
        """
        gauss = np.random.normal(mean, sigma, self.image.shape).astype(np.uint8)
        self.final_image = cv2.add(self.image, gauss)
        
        return self.final_image
        
    def save(self, filename: str) -> None:
        """
        Saves the image with Gaussian noise using the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = GaussianNoise()

    final_image = original_image.add_gaussian_noise()
    original_image.save('GaussianNoise')

    # Display original and processed images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()