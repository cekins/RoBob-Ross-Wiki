import sys
from image_functions import *
import cv2

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'city.jpg'
    else:
        filename = sys.argv[1]
    if len(sys.argv) < 3:
        max_pixels = 150
    else:
        max_pixels = int(sys.argv[2])
    res, img = get_strokes(filename, max_pixels)
    img2 = apply_stroke_pattern(res, img)
    cv2.imshow('image', img)
    cv2.waitKey(0)

