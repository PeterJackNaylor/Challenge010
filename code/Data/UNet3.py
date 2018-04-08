from optparse import OptionParser
import os
from glob import glob
from skimage.io import imread, imsave
import numpy as np
from utils.random_utils import CheckOrCreate
import pdb
from patch_img import Contours
def GenerateGT(input, output):
    files_GT = os.path.join(input, "GT_*", "*.png")
    for gt_path in glob(files_GT):
        out_path = gt_path.replace(input, output)
        yield gt_path, out_path

def PutContoursTo2(gt, size=3):
    gt = imread(gt)
    if gt.shape > 2:
        gt = gt[:,:,0]
    gt[gt > 0] = 1
    contours = Contours(gt, size)
    gt[contours > 0] = 2
    return gt

def save_path(output, rgb, original):
    out = rgb.replace(original, output)
    f = os.path.abspath(os.path.join(out, os.pardir))
    
    return out

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    (options, args) = parser.parse_args()
    #CheckOrCreate(options.output)
    train = True if options.train == "0" else False
    for gt_, out_ in GenerateGT(options.input, options.output):
        CheckOrCreate(os.path.dirname(out_))
        result = PutContoursTo2(gt_)
        imsave(out_, result)

    print options.train, options.input, options.norm