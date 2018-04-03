# HistoNormalization2.py

from optparse import OptionParser
import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
from utils.random_utils import CheckOrCreate
import shutil
import skimage.morphology
from HistogramNormalization2 import normalize_multi_channel

import pdb

def preprocessing_ellipse_tophat(img, ellipse_lambda=0.2):
    if len(img.shape) > 2:
        imin = img[:,:,0].copy()
    else: 
        imin = img.copy()
    total_area = imin.shape[0] * imin.shape[1]

    im_ellipse_open = skimage.morphology.ellipse_filter(imin, ellipse_lambda,
                                                        area_low_thresh=100, 
                                                        area_high_thresh=8000,
                                                        method='cut_first')
    im_ellipse_tophat = imin - im_ellipse_open

    return im_ellipse_tophat

def preprocessing_ellipse_filter(img, ellipse_lambda=0.82):
    if len(img.shape) > 2:
        imin = img[:,:,0].copy()
    else: 
        imin = img.copy()
    total_area = imin.shape[0] * imin.shape[1]

    im_ellipse_open = skimage.morphology.ellipse_filter(imin, ellipse_lambda,
                                                        area_low_thresh=100, 
                                                        area_high_thresh=8000,
                                                        method='direct')
    return im_ellipse_open



if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--processing_type', dest="processing_type", type="str")

    (options, args) = parser.parse_args()

    input_folder = options.input 
    if not os.path.isdir(input_folder):
        raise ValueError("folder %s does not exist" % input_folder)
    output_folder = options.output
    CheckOrCreate(output_folder)

    processing_type = options.processing_type

    # PARAMETERS
    ellipse_tophat_lambda=0.2
    ellipse_filter_lambda=0.82

    for root, dirs, files in os.walk(input_folder):        
        local_output_folder = root.replace(input_folder, output_folder)
        CheckOrCreate(local_output_folder)
        for filename in files:
            if filename[0] == ".":
                continue
            if filename.endswith('_mask.png'):
                continue
            imin = skimage.io.imread(os.path.join(root, filename))
            #try:
            #    imnorm = normalize_multi_channel(imin)
            #except: 
            #    pdb.set_trace()
            if processing_type == 'ellipse_tophat':
                imout = preprocessing_ellipse_tophat(imin, ellipse_tophat_lambda)
            elif processing_type == 'ellipse_filter':
                imout = preprocessing_ellipse_filter(imin, ellipse_filter_lambda)
            else: 
                raise ValueError("this pre-processing type has not yet been defined.")

            skimage.io.imsave(os.path.join(local_output_folder, filename), imout)

    print 'DONE'


