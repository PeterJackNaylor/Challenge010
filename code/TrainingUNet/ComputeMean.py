from optparse import OptionParser
from skimage.io import imread
from utils.FileCollector import GatherFiles
import numpy as np
import pdb

def ComputeMean(img_path):
    img = imread(img_path)
    if img.shape[2] == 4:
        if np.std(img[:,:,3]) < 1 :
            img = img[:,:,0:3]
    return np.mean(img, axis=(0, 1))

if __name__== "__main__":
    parser = OptionParser()
    parser.add_option("--input", dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    (options, args) = parser.parse_args()

    list_img, dic = GatherFiles(options.input, 10000, "train")
    ## hack to get all input files, exepct if there is 10000 folds...
    res = [ComputeMean(el) for el in list_img]
    mean_res = np.mean(res, axis=0)
    np.save(options.output, mean_res)
