import cv2 
import numpy as np

from common import read_img, save_img

class KernelFilter:
    def __init__(self) -> None:
        """
        Initializes the KernelFilter class and loads the input image.
        """
        self.image: np.ndarray = read_img()  
        self.final_image: np.ndarray = None   

    def apply_gaussian_filter(self, kernel_size: tuple[int, int] = (5, 5), sigma_x: float = 0.0) -> np.ndarray:
        """
        Apply Gaussian filter to the image.
        
        Args:
            kernel_size (tuple): Size of the kernel (must be odd).
            sigma_x (float): Standard deviation in the x direction.

        Returns:
            numpy.ndarray: Filtered image.
        """
        self.final_image = cv2.GaussianBlur(self.image, kernel_size, sigma_x)
        return self.final_image

    def apply_median_filter(self, kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter to the image.
        
        Args:
            kernel_size (int): Size of the kernel (must be odd).

        Returns:
            numpy.ndarray: Filtered image.
        """
        self.final_image = cv2.medianBlur(self.image, kernel_size)
        return self.final_image

    def apply_bilateral_filter(self, d: int = 9, sigma_color: float = 75.0, sigma_space: float = 75.0) -> np.ndarray:
        """
        Apply bilateral filter to the image.
        
        Args:
            d (int): Diameter of the pixel neighborhood.
            sigma_color (float): Standard deviation in color space.
            sigma_space (float): Standard deviation in coordinate space.

        Returns:
            numpy.ndarray: Filtered image.
        """
        self.final_image = cv2.bilateralFilter(self.image, d, sigma_color, sigma_space)
        return self.final_image
        
    def save(self, filename: str) -> None:
        """
        Saves the filtered image with the specified filename.
        
        Args:
            filename (str): The output filename prefixed with 'violet_'.
        """
        save_img(save_img_name='violet_' + filename, img=self.final_image)

if __name__ == "__main__":
    original_image = KernelFilter()

    final_image = original_image.apply_bilateral_filter()
    final_image = original_image.apply_gaussian_filter()
    final_image = original_image.apply_median_filter()
    original_image.save('KernelFilter')

    # Display original and processed images
    cv2.imshow('Original Image', original_image.image)
    cv2.imshow('After Image', final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()