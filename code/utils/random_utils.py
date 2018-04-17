import os
import numpy as np
from skimage.morphology import dilation, erosion, square, disk
from ImageTransform import flip_horizontal, flip_vertical
import pdb

def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

        
def CheckExistants(path):
    assert os.path.isdir(path)


def CheckFile(path):
    assert os.path.isfile(path)

def generate_wsl(ws):
    """
    to remove 
    """
    se = square(3)
    ero = ws.copy()
    ero[ero == 0] = ero.max() + 1
    ero = erosion(ero, se)
    ero[ws == 0] = 0

    grad = dilation(ws, se) - ero
    grad[ws == 0] = 0
    grad[grad > 0] = 255

    grad = grad.astype(np.uint8)
    return grad

def sliding_window(image, stepSize, windowSize):
    # slide a window across the imag
    for x in xrange(0, image.shape[0] - windowSize[0] + stepSize[0], stepSize[0]):
        for y in xrange(0, image.shape[1] - windowSize[1] + stepSize[1], stepSize[1]):
            # yield the current window  
            res_img = image[x:x + windowSize[0], y:y + windowSize[1]]
            change = False
            if res_img.shape[0] != windowSize[0]:
                x = image.shape[0] - windowSize[0]
                change = True
            if res_img.shape[1] != windowSize[1]:
                y = image.shape[1] - windowSize[1]
                change = True
            if change:
                res_img = image[x:x + windowSize[0], y:y + windowSize[1]]
            yield (x, y, x + windowSize[0], y + windowSize[1], res_img)


def UNetAugment(img):
    dim = img.shape
    i = 0
    new_dim = ()
    for c in dim:
        if i < 2:
            ne = c + 184
        else:
            ne = c
        i += 1
        new_dim += (ne, )

    result = np.zeros(shape=new_dim, dtype=img.dtype)
    n = 92
#       not right place to check for size..
#        assert CheckNumberForUnet(
#            dim[0] + 2 * n), "Dim not suited for UNet, it will create a wierd net"
    # middle
    result[n:-n, n:-n] = img.copy()
    # top middle
    result[0:n, n:-n] = flip_horizontal(result[n:(2 * n), n:-n])
    # bottom middle
    result[-n::, n:-n] = flip_horizontal(result[-(2 * n):-n, n:-n])
    # left whole
    result[:, 0:n] = flip_vertical(result[:, n:(2 * n)])
    # right whole
    result[:, -n::] = flip_vertical(result[:, -(2 * n):-n])
    #result = result.astype("uint8")
    return result

def expand_sym_padding(img, size, direction):
    assert direction in ["Bottom", "Right"]
    x, y = img.shape[:2]
    new_shape = np.array(img.shape)

    if direction in ["Bottom"]:
        new_shape[0] += size
        res = np.zeros(shape=tuple(new_shape), dtype=img.dtype)
        res[0:x] = img
        res[x:(x+size)] = flip_horizontal(img[-size:])
    else:
        new_shape[1] += size
        res = np.zeros(shape=tuple(new_shape), dtype=img.dtype)
        res[:, 0:y] = img
        res[:, y:(y+size)] = flip_vertical(img[:, -size:])
    return res

def UNetAdjust(img):
    x, y = img.shape[0:2]
    reste_x = x % 16
    if reste_x > 4:
        remove_x_pixel = -(reste_x - 4)
    elif reste_x < 4:
        remove_x_pixel = -(reste_x + 12)
    else:
        remove_x_pixel = None
    reste_y = y % 16
    if reste_y > 4:
        remove_y_pixel = -(reste_y - 4)
    elif reste_y < 4:
        remove_y_pixel = -(reste_y + 12)
    else:
        remove_y_pixel = None
    return img[0:remove_x_pixel, 0:remove_y_pixel]

def UNetAdjust_pixel(img):
    x, y = img.shape[0:2]
    reste_x = x % 16
    if reste_x > 4:
        remove_x_pixel = -(reste_x - 4)
    elif reste_x < 4:
        remove_x_pixel = -(reste_x + 12)
    else:
        remove_x_pixel = 0
    reste_y = y % 16
    if reste_y > 4:
        remove_y_pixel = -(reste_y - 4)
    elif reste_y < 4:
        remove_y_pixel = -(reste_y + 12)
    else:
        remove_y_pixel = 0
    return x + remove_x_pixel, y + remove_y_pixel

def color_bin(bin_labl):
    """
    Colors bin image so that nuclei come out nicer.
    """
    dim = bin_labl.shape
    x, y = dim[0], dim[1]
    res = np.zeros(shape=(x, y, 3))
    for i in range(1, bin_labl.max() + 1):
        rgb = np.random.normal(loc = 125, scale=100, size=3)
        rgb[rgb < 0 ] = 0
        rgb[rgb > 255] = 255
        rgb = rgb.astype(np.uint8)
        res[bin_labl == i] = rgb
    return res.astype(np.uint8)

def add_contours(rgb_image, contour, ds = 2):
    """
    Adds contours to images.
    The image has to be a binary image 
    """
    rgb = rgb_image.copy()
    contour[contour > 0] = 1
    boundery = contour - erosion(contour, disk(ds))
    rgb[boundery > 0] = np.array([0, 0, 0])
    return rgb

