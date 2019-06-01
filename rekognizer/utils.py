import requests

import cv2
import numpy as np
    

def read_image(image_url: str) -> np.array:
    res = requests.get(image_url)
    image = np.asarray(bytearray(res.content), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Return RGB image
    return image[..., ::-1]


def normalize_image(image: np.array):
    mean = np.mean(image)
    std = np.std(image)
    std_adj = np.maximum(std, 1.0 / np.sqrt(image.size))
    y = np.multiply(np.subtract(image, mean), 1 / std_adj)

    return y
