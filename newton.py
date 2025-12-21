from numba import njit, prange
import numpy as np
from PIL import Image
import colorsys

MAX_ITERATIONS = 50
TOLERANCE = 0.0001
MODE = 4
START_COLOR, END_COLOR = (0, 0, 0), (255, 255, 255)
TRANSITION_SPEED = 1.2
RESOLUTION = 1000
HALF_RES = RESOLUTION // 2
WIDTH = 1
INCREMENT = WIDTH / RESOLUTION
CENTER = (1.5, 2.28)
BACKGROUND_IMAGE = "charcoal.png"
FILENAME = "samplefractal.png"



# Polynomials f(z) and f'(z)
@njit
def f(z): return 8*z**7 + z**6 + 3*z**5 + 7*z**4 + 2*z**3 + 4*z**2 + 5*z + 6
@njit
def fp(z): return 56*z**6 + 6*z**5 + 15*z**4 + 28*z**3 + 6*z**2 + 8*z + 5

# Roots of the polynomial and the number of roots it has
coefficients = [8, 1, 3, 7, 2, 4, 5, 6]
roots = np.roots(coefficients)
del coefficients
root_num = len(roots)

@njit
def root_find(start_point):
    """
    Takes in a point on the complex plane and
    returns which root of the polynomial it is
    closest to after MAX_ITERATIONS iterations
    of Newton's method.

    Args:
        start_point (complex): Starting point
        for Newton's method.

    Returns:
        int: Integer representing one of the
        roots of the polynomial.
    """
    z = start_point
    for i in range(MAX_ITERATIONS):
        deriv = fp(z)
        if deriv == 0: break
        z -= f(z) / deriv
    root_index = 0
    min_dist = 0
    for index in range(root_num):
        if index == 0:
            root_index = index
            min_dist = abs(roots[index] - z)
        elif abs(roots[index] - z) < min_dist:
            root_index = index
            min_dist = abs(roots[index] - z)
    return root_index

@njit
def speed_find(start_point):
    """
    Takes in a point on the complex plane and
    returns the speed at which that point
    converges to a root of the polynomial.
    Speed is based on how many iterations of
    Newton's method it took for the point to
    be within a desired tolerance of any root,
    lower values indicate faster convergence.

    Args:
        start_point (complex): Starting point
        for Newton's method.

    Returns:
        float: Float value between 0 and 1,
        0 means the starting point is already
        at a root, 1 means the starting point
        did not converge to a root after
        MAX_ITERATIONS of Newton's method.
    """
    z = start_point
    for i in range(MAX_ITERATIONS):
        deriv = fp(z)
        if deriv == 0: break
        z -= f(z) / deriv
        for index in range(root_num):
            if abs(roots[index] - z) < TOLERANCE:
                return i / MAX_ITERATIONS
    return 1

@njit(parallel=True)
def get_pixel_roots():
    """
    Maps each image pixel to a point on the
    complex plane and finds the root it is
    closest to after MAX_ITERATIONS iterations
    of Newton's method.

    Returns:
        ndarray: NxN matrix containing the
        root that each pixel converges to.
    """
    pixel_roots = np.zeros((RESOLUTION, RESOLUTION), dtype=np.int32)
    for i in prange(RESOLUTION):
        for j in prange(RESOLUTION):
            ii = i - HALF_RES
            jj = HALF_RES - j
            point = complex(ii * INCREMENT + CENTER[0], jj * INCREMENT + CENTER[1])
            pixel_roots[i, j] = root_find(point)
    return pixel_roots

@njit(parallel=True)
def get_pixel_convergence_speeds():
    """
    Maps each image pixel to a point on the
    complex plane and finds the speed at
    which that point converges to a root of
    the polynomial.

    Returns:
        ndarray: NxN matrix containing the
        speed that each pixel converges at.
    """
    pixel_convergence_speeds = np.zeros((RESOLUTION, RESOLUTION))
    for i in prange(RESOLUTION):
        for j in prange(RESOLUTION):
            ii = i - HALF_RES
            jj = HALF_RES - j
            point = complex(ii * INCREMENT + CENTER[0], jj * INCREMENT + CENTER[1])
            pixel_convergence_speeds[i, j] = speed_find(point)
    return pixel_convergence_speeds

def root_to_hue(root_id):
    """
    Takes in a root index, divides it by the
    total number of roots for the given polynomial,
    and uses the result as the hue for a
    color.

    Args:
        root_id (int): Root index

    Returns:
        int: Red value between 0 and 255
        int: Green value between 0 and 255
        int: Blue value between 0 and 255
    """
    r, g, b = colorsys.hsv_to_rgb((root_id / root_num), 0.8, 0.8)
    return int(r * 255), int(g * 255), int(b * 255)

@njit
def root_to_rgb(root_id, start_color, end_color):
    """
    Interpolates an RGB color between two
    endpoints based on the given root index.

    Args:
        root_id (int): Root index
        start_color (tuple): RGB color to start at
        end_color (tuple): RGB color to end at

    Returns:
        tuple: RGB value between start_color
        and end_color
    """
    r = start_color[0] + (end_color[0] - start_color[0]) * (root_id / root_num) ** TRANSITION_SPEED
    g = start_color[1] + (end_color[1] - start_color[1]) * (root_id / root_num) ** TRANSITION_SPEED
    b = start_color[2] + (end_color[2] - start_color[2]) * (root_id / root_num) ** TRANSITION_SPEED
    return round(r), round(g), round(b)

@njit
def speed_to_rgb(speed, start_color, end_color):
    """
    Interpolates an RGB color between two
    endpoints based on the given convergence
    speed.

    Args:
        speed (float): Convergence speed
        between 0 and 1
        start_color (tuple): RGB color to start at
        end_color (tuple): RGB color to end at

    Returns:
        tuple: RGB value between start_color and end_color
    """
    r = start_color[0] + (end_color[0] - start_color[0]) * speed ** TRANSITION_SPEED
    g = start_color[1] + (end_color[1] - start_color[1]) * speed ** TRANSITION_SPEED
    b = start_color[2] + (end_color[2] - start_color[2]) * speed ** TRANSITION_SPEED
    return round(r), round(g), round(b)

def speed_to_inverse(speed, color):
    """
    Interpolates an RGB color between the
    given RGB value and its inverse based
    on the given convergence speed.

    Args:
        speed (float): Convergence speed
        between 0 and 1
        color (tuple): Base RGB color

    Returns:
        tuple: RGB value between color and the
        inverse of color
    """
    r = speed ** TRANSITION_SPEED * (255 - color[0]) + (1 - speed ** TRANSITION_SPEED) * color[0]
    g = speed ** TRANSITION_SPEED * (255 - color[1]) + (1 - speed ** TRANSITION_SPEED) * color[1]
    b = speed ** TRANSITION_SPEED * (255 - color[2]) + (1 - speed ** TRANSITION_SPEED) * color[2]
    return round(r), round(g), round(b)

def create_image():
    """
    Creates a new image and colors each
    pixel based on the chosen mode. Saves
    the created image as a .png file in the
    working directory.
    Mode 1:
        Colors each pixel based on which
        root it will converge to, hues evenly
        spaced around the color wheel.
    Mode 2:
        Colors each pixel based on which
        root it will converge to, colors
        interpolated between the starting
        color and ending color.
    Mode 3:
        Colors each pixel based on convergence
        speed, colors interpolated between the
        starting color and ending color.
    Mode 4:
        Colors each pixel based on convergence
        speed, colors interpolated between the
        color of the pixel in the background
        image at that same point and the inverse
        of that color.
    """
    im = Image.new(mode='RGB', size=(RESOLUTION, RESOLUTION))
    pixels = im.load()
    if MODE == 1:
        pixel_roots = get_pixel_roots()
        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                pixels[i, j] = (root_to_hue(pixel_roots[i, j]))
    elif MODE == 2:
        pixel_roots = get_pixel_roots()
        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                pixels[i, j] = (root_to_rgb(pixel_roots[i, j], START_COLOR, END_COLOR))
    elif MODE == 3:
        pixel_convergence_speeds = get_pixel_convergence_speeds()
        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                pixels[i, j] = (speed_to_rgb(pixel_convergence_speeds[i, j], START_COLOR, END_COLOR))
    elif MODE == 4:
        pixel_convergence_speeds = get_pixel_convergence_speeds()
        image = Image.open(BACKGROUND_IMAGE).resize((RESOLUTION, RESOLUTION), Image.Resampling.LANCZOS).convert('RGB')
        for i in range(RESOLUTION):
            for j in range(RESOLUTION):
                rgb = image.getpixel((i, j))
                pixels[i, j] = (speed_to_inverse(pixel_convergence_speeds[i, j], rgb))
    im.save(FILENAME)

if __name__ == '__main__':
    print("Generating new image...")
    create_image()
    print("Done.")