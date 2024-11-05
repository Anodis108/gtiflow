import cv2 
import numpy as np

from . import read_img, save_img

class SaltAndPepperNoise:
    """
    Class for adding salt and pepper noise to an image.
    
    Attributes:
        image (numpy.ndarray): Input image.
        final_image (numpy.ndarray): Image with added noise.
    """
    
    def __init__(self) -> None:
        """
        Initializes the SaltAndPepperNoise class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def add_noise(self, salt_prob: float = 0.02, pepper_prob: float = 0.02) -> np.ndarray:
        """
        Adds salt and pepper noise to the image.
        
        Args:
            salt_prob (float): Probability of salt noise (value 1).
            pepper_prob (float): Probability of pepper noise (value 0).
        
        Returns:
            numpy.ndarray: Image with added salt and pepper noise.
        """
        noisy_image = np.copy(self.image)

        # Calculate the number of salt and pepper pixels
        total_pixels = noisy_image.size
        num_salt = int(total_pixels * salt_prob)
        num_pepper = int(total_pixels * pepper_prob)

        # Add salt noise
        coords = [np.random.randint(0, i - 1, num_salt) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1], :] = 1  # Salt value is 1 (white)

        # Add pepper noise
        coords = [np.random.randint(0, i - 1, num_pepper) for i in noisy_image.shape]
        noisy_image[coords[0], coords[1], :] = 0  # Pepper value is 0 (black)

        self.final_image = noisy_image
        return self.final_image

    def save(self, filename: str) -> None:
        """
        Saves the image with noise using the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = SaltAndPepperNoise()

    final_image = original_image.add_noise()
    original_image.save('SaltAndPepperNoise')

    # Display original and noisy images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()