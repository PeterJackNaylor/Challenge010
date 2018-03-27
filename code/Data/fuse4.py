from optparse import OptionParser
import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np
from utils.random_utils import CheckOrCreate
import pdb
def GenerateRGB_Norm(original, norm, train):
    if train:
        files_rgb = os.path.join(original, "Slide_*", "*.png")
    else:
        files_rgb = os.path.join(original, "*", "images", "*.png")
    for rgb_path in glob(files_rgb):
        norm_path = rgb_path.replace(original, norm)
        yield rgb_path, norm_path

def fuse(rgb, greyscale):
    rgb = imread(rgb)[:,:,0:3]
    greyscale = imread(greyscale)[:,:,0]
    x, y = greyscale.shape
    fused = np.zeros(shape=(x, y, 4), dtype="uint8")
    fused[:,:,0:3] = rgb
    fused[:,:,3] = greyscale
    return fused

def save_path(output, rgb, original):
    out = rgb.replace(original, output)
    f = os.path.abspath(os.path.join(out, os.pardir))
    CheckOrCreate(f)
    return out

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--train', dest="train", type="str")
    parser.add_option('--original', dest="original", type="str")
    parser.add_option('--norm', dest="norm", type="str")
    parser.add_option('--output', dest="output", type="str")
    (options, args) = parser.parse_args()
    #CheckOrCreate(options.output)
    train = True if options.train == "0" else False
    for rgb_, grey_ in GenerateRGB_Norm(options.original, options.norm, train):
        result = fuse(rgb_, grey_)
        save_p = save_path(options.output, rgb_, options.original)
        imsave(save_p, result)

    print options.train, options.original, options.norm