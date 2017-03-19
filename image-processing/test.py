import cv2
import numpy as np
from copy import deepcopy

MAX_PIXELS = 500

COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "gray": (128, 128, 128),
    "lime": (0, 255, 0),
    "green": (0, 128, 0),
    "purple": (128, 0, 128),
    "navy": (0, 0, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "cyan": (0, 255, 255),
    "teal": (0, 128, 128),
    "orange": (265, 165, 0)
}

def get_distance(color1, color2):
    dsq = 0
    for i in range(3):
        dsq += np.power((color1[i] - color2[i]), 2)
    return np.sqrt(dsq)

def get_nearest_color(pixel):
    min_color = "red"
    min_color_dist = float("inf")
    for color in COLORS:
        value = COLORS[color]
        d = get_distance(pixel, value)
        if d < min_color_dist:
            min_color_dist = d
            min_color = color
    return min_color


def get_output_size(image):
    height = image.shape[0]
    width = image.shape[1]
    max_dim = max(height, width)
    if max_dim <= MAX_PIXELS:
        return cv2.GetSize(image)
    scale_factor = MAX_PIXELS / float(max_dim)
    return int(round(width * scale_factor)), int(round(height* scale_factor))


def get_average_color(image, i, j):
    neighbors = 0
    res = [0.0, 0.0, 0.0]
    for ii in range(-1, 2):
        if 0 <= i + ii < image.shape[0]:
            for jj in range(-1, 2):
                if 0 <= j + jj < image.shape[1]:
                    neighbors += 1
                    res[0] += image[i + ii, j + jj, 0]
                    res[1] += image[i + ii, j + jj, 1]
                    res[2] += image[i + ii, j + jj, 2]
    return (int(round(res[0] / neighbors)), int(round(res[1] / neighbors)), int(round(res[2] / neighbors)))



def transform_image(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            color = get_nearest_color(image[i, j])
            image[i, j] = COLORS[color]



if __name__ == '__main__':
    img = cv2.imread('fruit.jpg')
    dst_size = get_output_size(img)
    dst = cv2.resize(img, dst_size)
    edges = cv2.Canny(dst, 100, 200)
    transform_image(dst)
    cv2.imshow('image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
