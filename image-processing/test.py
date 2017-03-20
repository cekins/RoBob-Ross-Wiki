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
    return COLORS[min_color]


def get_output_size(image):
    height = image.shape[0]
    width = image.shape[1]
    max_dim = max(height, width)
    if max_dim <= MAX_PIXELS:
        return image.shape[0], image.shape[1]
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



def reduce_colors(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = get_nearest_color(image[i, j])

def white_2_black(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.array_equal(image[i, j], [255, 255, 255]):
                image[i, j] = [0, 0, 0]

def get_black_image(image):
    return np.zeros((image.shape[0], image.shape[1]), np.uint8)

def isolate_color(image, color):
    res = get_black_image(image)
    color_pixel = COLORS[color]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.array_equal(image[i, j], color_pixel):
                res[i, j] = 255
    return res

def find_color_contours(image):
    res = get_black_image(image) 
    for color in COLORS:
        tmp = isolate_color(image, color)
        im2, contours, heirarchy = cv2.findContours(tmp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        for i in range(len(contours)):
            cv2.drawContours(res, contours, i, 255)
    return res

def get_cc_points(labels):
    connected_components = [[] for i in range(len(labels))]
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            connected_components[labels[i, j]].append((i, j))
    return connected_components

def contains(cc_1, cc_2):
    for i in cc_1:
        for j in cc_2:
            if cmp(i, j):
                return True
    return False

if __name__ == '__main__':
    img = cv2.imread('city.jpg')
    dst_size = get_output_size(img)
    img = cv2.resize(img, dst_size)
    orig = deepcopy(img)

    img = cv2.GaussianBlur(img, (0, 0), 1)
    reduce_colors(img)
    tmp = find_color_contours(img)
    output = cv2.connectedComponentsWithStats(tmp)
    stats = output[2]
    labels = output[1]
    cc_points = get_cc_points(labels)
    cc_points.sort(key=len, reverse=True)
    adj_mat = [[0] * len(labels) for i in range(len(labels))]
    for i in range(len(cc_points) - 1):
        for j in range(i + 1, len(cc_points)):
            if contains(cc_points[i], cc_points[j]):
                adj_mat[i][j] = 1
    print sum([sum(i) for i in adj_mat])
    cv2.imshow('contours', tmp)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
