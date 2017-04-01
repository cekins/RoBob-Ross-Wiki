import cv2
import numpy as np
from copy import deepcopy
import cProfile
import Queue
from pprint import pprint
import sys
import json

COLORS = {
    'red': (255, 0, 0),
    'blue': (0, 0, 255),
    'yellow': (255, 255, 0),
    'gray': (128, 128, 128),
    'lime': (0, 255, 0),
    'green': (0, 128, 0),
    'purple': (128, 0, 128),
    'navy': (0, 0, 128),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'cyan': (0, 255, 255),
    'teal': (0, 128, 128),
    'orange': (255, 165, 0)
}

def get_color_from_pixel(pixel):
    for color in COLORS:
        if np.array_equal(pixel, COLORS[color]):
            return color
    return color

# gets the Euclidian distance between two colors
def get_distance(color1, color2):
    dsq = 0
    for i in range(3):
        dsq += np.power((color1[i] - color2[i]), 2)
    return np.sqrt(dsq)

# gets the closest color in the COLORS dictionary to a pixel in the image
def get_nearest_color(pixel):
    min_color = 'red'
    min_color_dist = float('inf')
    for color in COLORS:
        value = COLORS[color]
        d = get_distance(pixel, value)
        if d < min_color_dist:
            min_color_dist = d
            min_color = color
    return COLORS[min_color]

# Gets the size of the output image, 500px max in either direction
def get_output_size(image, max_pixels):
    height = image.shape[0]
    width = image.shape[1]
    max_dim = max(height, width)
    if max_dim <= max_pixels:
        return image.shape[0], image.shape[1]
    scale_factor = max_pixels / float(max_dim)
    return int(round(width * scale_factor)), int(round(height* scale_factor))

# Not in use
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


# Change each pixel in the image to the nearest color in COLORS based on euclidian distance
def reduce_colors(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            image[i, j] = get_nearest_color(image[i, j])

# change white pixels to black in the image
def white_2_black(image):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.array_equal(image[i, j], [255, 255, 255]):
                image[i, j] = [0, 0, 0]

# Get an all-black single-channel image with the same dimensions as input
def get_black_image(image):
    return np.zeros((image.shape[0], image.shape[1]), np.uint8)

# Get an all-black 3-channel image with same dimensions as input
def get_black_color_image(image):
    return np.zeros(image.shape, np.uint8)

# Get a single-channel  image with the same dimensions as input, where all colors that are the input color 
# are white, else black
def isolate_color(image, color):
    res = get_black_image(image)
    color_pixel = COLORS[color]
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if np.array_equal(image[i, j], color_pixel):
                res[i, j] = 255
    return res

# Get a list of white points in an image. Used to isolate each connected connected components
def get_white_points(image):
    ccs = cv2.connectedComponentsWithStats(image)
    centroids = np.uint32(ccs[3].round())
    centroid = (-1, -1)
    for c in centroids:
        if image[c[1], c[0]] == 255:
            centroid = (c[1], c[0])
    if np.array_equal(centroid, [-1, -1]):
        return []
    q = Queue.Queue()
    q.put(centroid)
    res = []
    visited = get_black_image(image)
    visited[centroid] = 255
    while not q.empty():
        cur = q.get()
        res.append(cur)
        neighbors = []
        for i in range(-1, 2):
            if 0 <= cur[0] + i < image.shape[0]:
                for j in range(-1, 2):
                    if 0 <= cur[1] + j < image.shape[1]:
                        tmp = (cur[0] + i, cur[1] + j)
                        if image[tmp] == 255 and visited[tmp] == 0:
                            visited[tmp] = 255
                            q.put((cur[0] + i, cur[1] + j))
    return res

def get_neighbors(coord, shape):
    neighbors = []
    for i in range(-1, 2):
        if 0 <= coord[0] + i < shape[0]:
            for j in range(-1, 2):
                if 0 <= coord[1] + j < shape[1]  and not (i == j and i == 0):
                    neighbors.append((coord[0] + i, coord[1] + j))
    return neighbors


def find_color_regions(image):
    res = {}
    visited = np.zeros(image.shape[:2], np.uint8)
    q = Queue.Queue()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if visited[i, j] == 255 or image[i, j] == 0:
                continue
            visited[i, j] = 255
            q.put((i, j))
            region = []
            while not q.empty():
                cur = q.get()
                region.append(cur)
                neighbors = get_neighbors(cur, image.shape)
                for neighbor in neighbors:
                    if visited[neighbor] == 0 and image[neighbor] == 255:
                        visited[neighbor] = 255
                        q.put(neighbor)
            res.append(region)
    return res


def contains(cc_1, cc_2):
    for i in cc_1:
        for j in cc_2:
            if cmp(i, j):
                return True
    return False

# get a dict mapping each color to a list of lists, where each internal list is a connected component color region
def restore_image(regions, image):
    res = get_black_color_image(image)
    for color in regions:
        pix_val = COLORS[color]
        for region in regions[color]:
            for pixel in region:
                res[pixel] = pix_val
    return res
            
def find_color_regions(image):
    res = []
    visited = np.zeros(image.shape[:2], np.uint8)
    q = Queue.Queue()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] == 0 or visited[i, j] == 255:
                continue
            visited[i, j] = 255
            q.put((i, j))
            region = []
            while not q.empty():
                cur = q.get()
                region.append(cur)
                neighbors = get_neighbors(cur, image.shape)
                for neighbor in neighbors:
                    if visited[neighbor] == 0 and image[neighbor] == 255:
                        visited[neighbor] = 255
                        q.put(neighbor)
            res.append(region)
    return res

def distance(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def path_dist(path):
    dist = 0
    for i in range(len(path) - 1):
        dist += distance(path[0], path[1])
    return dist

def rotate(l, n):
    return l[n:] + l[:n]

def two_opt(path, img):
    for i in range(len(path) - 2):
        for j in range(i + 1, len(path) - 1):
            if distance(path[i], path[i + 1]) + distance(path[j], path[j + 1]) > distance(path[i], path[j]) + distance(path[i + 1], path[j + 1]):
                tmp = path[i + 1]
                path[i + 1] = path[j]
                path[j] = tmp
                path[i + 2 : j] = reversed(path[i + 2:j])
    res = []
    for i in range(1, len(path)):
        if len(res) == 0 or distance(res[-1][-1], path[i]) >= 1.5:
            res.append([])
        res[-1].append(path[i])
    res = [x for x in res if len(x) >= 2]
    return res

def coord_to_cm(coord, shape):
    scale = 8.0 / max(shape)
    return (coord[0] * scale, coord[1] * scale)

    

def process(filename, max_pixels):
    img = cv2.imread(filename)
    dst_size = get_output_size(img, max_pixels)
    img = cv2.resize(img, dst_size)
    orig = deepcopy(img)
    img = cv2.GaussianBlur(img, (0, 0), 1.3)
    tmp = cv2.Canny(img, 255/3,  255)
    res = find_color_regions(tmp)
    edges = []
    for edge in res:
        edges.extend(two_opt(edge, tmp))
    res2 = []
    for edge in edges:
        edge = cv2.approxPolyDP(np.array(edge), max_pixels / 75.0, False)
        res2.append(edge)
    tmp2 = get_black_image(img)
    for i in range(len(res2)):
        for point in res2[i]:
            temp = point[0, 0]
            point[0, 0] = point[0, 1]
            point[0, 1] = temp
        cv2.polylines(tmp2, res2, False, 255)
    #cv2.imshow('edges', tmp)
    #cv2.imshow('simplified', tmp2)
    #cv2.waitKey(0)
    final = []
    for edge in res2:
        final.append([])
        for point in edge:
            final[-1].append(coord_to_cm(point[0], tmp2.shape))
    print json.dumps(final)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        filename = 'city.jpg'
    else:
        filename = sys.argv[1]
    if len(sys.argv) < 3:
        max_pixels = 150
    else:
        max_pixels = int(sys.argv[2])
    process(filename, max_pixels)
