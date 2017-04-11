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
        return image.shape[1], image.shape[0]
    scale_factor = max_pixels / float(max_dim)
    return int(round(width * scale_factor)), int(round(height * scale_factor))

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
                if 0 <= coord[1] + j < shape[1]:
                    neighbors.append((coord[0] + i, coord[1] + j))
    return neighbors

# Get a dictionary mapping each color to a list of regions of that color,
# where each region is a list of contiguous pixels that have that color.
def find_color_regions(image):
    res = {}
    visited = np.zeros(image.shape[:2], np.uint8)
    q = Queue.Queue()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if visited[i, j] == 255:
                continue
            visited[i, j] = 255
            q.put((i, j))
            color = get_color_from_pixel(image[i, j])
            if color not in res:
                res[color] = []
            region = []
            while not q.empty():
                cur = q.get()
                region.append(cur)
                neighbors = get_neighbors(cur, image.shape)
                for neighbor in neighbors:
                    if visited[neighbor] == 0 and np.array_equal(image[cur], image[neighbor]):
                        visited[neighbor] = 255
                        q.put(neighbor)
            res[color].append(region)
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
            
def get_grid_neighbors(point, shape, spacing):
    res = []
    if point[0] - spacing >= 0:
        res.append((point[0] - spacing, point[1]))
    if point[0] + spacing < shape[0]:
        res.append((point[0] + spacing, point[1]))
    if point[1] - spacing >= 0:
        res.append((point[0], point[1] - spacing))
    if point[1] + spacing < shape[1]:
        res.append((point[0], point[1] + spacing))
    return res

class GraphNode:
    def __init__(self, position, color):
        self.position = position
        self.color = color
        self.neighbors = []

class Graph:
    def __init__(self):
        self.nodes = []
        self.graph_dict = {}

    def __str__(self):
        res = ''
        for i in range(len(self.nodes)):
            res += str(self.nodes[i].position) + ': '
            for j in self.nodes[i].neighbors:
                res += str(self.nodes[j].position) + ', '
            res += '\n'
        return res

    def add_line(self, point1, point2, color):
        if point1 not in self.graph_dict:
            self.graph_dict[point1] = len(self.nodes)
            self.nodes.append(GraphNode(point1, color))
        if point2 not in self.graph_dict:
            self.graph_dict[point2] = len(self.nodes)
            self.nodes.append(GraphNode(point2, color))
        node1 = self.nodes[self.graph_dict[point1]]
        node2 = self.nodes[self.graph_dict[point2]]
        node1.neighbors.append(self.graph_dict[point2])
        node2.neighbors.append(self.graph_dict[point1])

    def rem_dups(self):
        for i in self.nodes:
            i.neighbors = list(set(i.neighbors))

    def dfs(self, current, visited, target, cycles):
        visited.append(self.nodes[current].position)
        if current == target and len(visited) == 4:
            cycles.append(deepcopy(visited))
            del visited[-1]
            return
        if len(visited) >= 4:
            del visited[-1]
            return
        k = 0
        for i in self.nodes[current].neighbors:
            if i == target:
                print len(visited)
            if self.nodes[i].position not in visited or (i == target and len(visited) == 3):   
                self.dfs(i, visited, target, cycles)
        del visited[-1]

    def get_cycles(self):
        cycles = {}
        for i in range(len(self.nodes)):
            node = self.nodes[i]
            if node.color not in cycles:
               cycles[node.color] = []
            these_cycles = []
            self.dfs(i, [], i, these_cycles)
            if len(these_cycles) > 0:
                cycles[node.color].append(these_cycles)
        return cycles

def draw_grid(blob, image, color, graph):
    color_name = get_color_from_pixel(color)
    black = get_black_image(image)
    for point in blob:
        black[point] = 255
    visted = deepcopy(black)
    spacing = max(black.shape) / 100
    points = []
    for pixel in blob:
        if pixel[0] % spacing == 0 and pixel[1] % spacing == 0:
            points.append(pixel)
    for point in points:
        neighbors = get_grid_neighbors(point, image.shape, spacing)
        for neighbor in neighbors:
            if black[neighbor] == 255:
                cv2.line(image, (point[1], point[0]),  (neighbor[1], neighbor[0]), color)
                graph.add_line(point, neighbor, color_name)



def process(filename, max_pixels):
    img = cv2.imread(filename)
    dst_size = get_output_size(img, max_pixels)
    img = cv2.resize(img, dst_size)
    orig = deepcopy(img)
    img = cv2.GaussianBlur(img, (0, 0), 1)
    reduce_colors(img)
    regions = find_color_regions(img)
    tmp = get_black_color_image(img)
    graph = Graph()
    for color in regions:
        for region in regions[color]:
            draw_grid(region, tmp, COLORS[color], graph)
    graph.rem_dups()
    cycles = graph.get_cycles()
    pprint(cycles)
    cv2.imshow('image', tmp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print json.dumps(regions)

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
