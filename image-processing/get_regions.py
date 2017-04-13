import sys
from image_functions import *

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'city.jpg'
    else:
        filename = sys.argv[1]
    if len(sys.argv) < 3:
        max_pixels = 150
    else:
        max_pixels = int(sys.argv[2])
    process_squares(filename, max_pixels)
