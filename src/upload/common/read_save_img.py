import cv2
import numpy as np  # Importing numpy for type hinting
from src.upload.common.constants import SAVE_IMAGE_PATH, IMAGE_PATH

def read_img(img_path: str = IMAGE_PATH) -> np.ndarray:
    """
    Read an image from the specified path.

    Args:
        img_path (str): Path to the image file. Default is `IMAGE_PATH`.

    Returns:
        numpy.ndarray: Image data if successful, or `None` if failed.
    """
    return cv2.imread(img_path)

def save_img(save_img_name: str, img: np.ndarray, save_img_path: str = SAVE_IMAGE_PATH) -> None:
    """
    Save an image with the specified name.

    Args:
        save_img_name (str): Name for the saved image (without extension).
        img (numpy.ndarray): Image data to save.
        save_img_path (str): Directory path to save the image. Default is `SAVE_IMAGE_PATH`.

    Returns:
        None
    """
    cv2.imwrite(r'{0}\{1}.jpg'.format(save_img_path, save_img_name), img)
    