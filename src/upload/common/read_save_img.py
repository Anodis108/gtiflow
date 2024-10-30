import cv2
from src.upload.common.constants import SAVE_IMAGE_PATH, IMAGE_PATH

def read_img(img_path=IMAGE_PATH):
    return cv2.imread(img_path)

def save_img(save_img_name, img, save_img_path=SAVE_IMAGE_PATH):
    cv2.imwrite(r'{0}\{1}.jpg'.format(save_img_path, save_img_name), img)
    