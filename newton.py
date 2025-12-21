import numba
import numpy as np
from numba import njit, prange
import PIL
import colorsys

MAX_ITERATIONS = 50
TOLERANCE = 0.0001
MODE = 3
COLOR_1, COLOR_2 = [0, 0, 0], [237, 201, 0]
TRANSITION_SPEED = 1.2
SIZE = 1000 #Adjust resolution of image
HALF_SIZE = SIZE // 2
INCREMENT = 1 / SIZE #Adjust width of frame
CENTER = (1.5, 2.28) #Adjust center of frame
FILENAME = ("sample fractal.png")



#x^8 + 15x^4 - 16
@njit
def f(z): return 8*z**7 + z**6 + 3*z**5 + 7*z**4 + 2*z**3 + 4*z**2 + 5*z + 6

@njit
def fp(z): return 56*z**6 + 6*z**5 + 15*z**4 + 28*z**3 + 6*z**2 + 8*z + 5

coefficients = [8, 1, 3, 7, 2, 4, 5, 6]
roots = np.roots(coefficients)
del coefficients
root_num = len(roots)

@njit
def root_find(start_point):
    z = start_point
    for i in range(MAX_ITERATIONS):
        deriv = fp(z)
        if deriv == 0: break
        z -= f(z) / deriv
    root_id = 0
    min_dist = 0
    for id in range(root_num):
        if id == 0:
            root_id = id
            min_dist = abs(roots[id] - z)
        elif abs(roots[id] - z) < min_dist:
            root_id = id
            min_dist = abs(roots[id] - z)
    return root_id

@njit
def speed_find(start_point):
    point = start_point
    for i in range(MAX_ITERATIONS):
        deriv = fp(point)
        if deriv == 0: break
        point -= f(point) / deriv
        for id in range(root_num):
            if abs(roots[id] - point) < TOLERANCE:
                return i / MAX_ITERATIONS
    return 1

@njit(parallel=True)
def pixel_roots():
    pixel_roots = np.zeros((SIZE, SIZE), dtype=np.int32)
    for i in prange(SIZE):
        for j in prange(SIZE):
            ii = i - HALF_SIZE
            jj = HALF_SIZE - j
            point = complex(ii * INCREMENT + CENTER[0], jj * INCREMENT + CENTER[1])
            pixel_roots[i, j] = root_find(point)
    return pixel_roots

@njit(parallel=True)
def pixel_convergence_speeds():
    pixel_convergence_speeds = np.zeros((SIZE, SIZE))
    for i in prange(SIZE):
        for j in prange(SIZE):
            ii = i - HALF_SIZE
            jj = HALF_SIZE - j
            point = complex(ii * INCREMENT + CENTER[0], jj * INCREMENT + CENTER[1])
            pixel_convergence_speeds[i, j] = speed_find(point)
    return pixel_convergence_speeds

def root_to_hsv(root):
    r, g, b = colorsys.hsv_to_rgb((root / root_num), 0.8, 0.8)
    return int(r * 255), int(g * 255), int(b * 255)

@njit
def root_to_rgb(root, start_rgb, end_rgb):
    r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * (root / root_num)
    g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * (root / root_num)
    b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * (root / root_num)
    return round(r), round(g), round(b)

@njit
def speed_to_rgb(speed, start_rgb, end_rgb):
    r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * speed ** TRANSITION_SPEED
    g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * speed ** TRANSITION_SPEED
    b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * speed ** TRANSITION_SPEED
    return round(r), round(g), round(b)


def speed_to_inverse(speed, rgb):
    r = speed ** TRANSITION_SPEED * (255 - rgb[0]) + (1 - speed ** TRANSITION_SPEED) * rgb[0]
    g = speed ** TRANSITION_SPEED * (255 - rgb[1]) + (1 - speed ** TRANSITION_SPEED) * rgb[1]
    b = speed ** TRANSITION_SPEED * (255 - rgb[2]) + (1 - speed ** TRANSITION_SPEED) * rgb[2]
    return round(r), round(g), round(b)

def create_image():
    im = PIL.Image.new(mode='RGB', size=(SIZE, SIZE))
    pixels = im.load()
    if MODE == 1:
        pixel_root = pixel_roots()
        for i in range(SIZE):
            for j in range(SIZE):
                pixels[i, j] = (root_to_hsv(pixel_root[i, j]))
    elif MODE == 2:
        pixel_root = pixel_roots()
        for i in range(SIZE):
            for j in range(SIZE):
                pixels[i, j] = (root_to_rgb(pixel_root[i, j], COLOR_1, COLOR_2))
    elif MODE == 3:
        pixel_convergence_speed = pixel_convergence_speeds()
        for i in range(SIZE):
            for j in range(SIZE):
                pixels[i, j] = (speed_to_rgb(pixel_convergence_speed[i, j], COLOR_1, COLOR_2))
    elif MODE == 4:
        pixel_convergence_speed = pixel_convergence_speeds()
        image = PIL.Image.open('charcoal.jpg').resize((SIZE, SIZE), PIL.Image.Resampling.LANCZOS).convert('RGB')
        for i in range(SIZE):
            for j in range(SIZE):
                rgb = image.getpixel((i, j))
                pixels[i, j] = (speed_to_inverse(pixel_convergence_speed[i, j], rgb))
    im.show()
    '''im.save(FILENAME)'''

create_image()