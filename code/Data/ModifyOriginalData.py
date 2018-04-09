from optparse import OptionParser
import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
from utils.random_utils import CheckOrCreate
from scipy.stats import skew
import shutil
from HistogramNormalization2 import normalize_multi_channel
import pdb

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    (options, args) = parser.parse_args()

    input_folder = options.input 
    if not os.path.isdir(input_folder):
        raise ValueError("folder %s does not exist" % input_folder)
    output_folder = options.output
    CheckOrCreate(output_folder)

    for root, dirs, files in os.walk(input_folder):        
        local_output_folder = root.replace(input_folder, output_folder)
        CheckOrCreate(local_output_folder)
        #pdb.set_trace()
        if "images" in root:
            for filename in files:
                if filename[0] == ".":
                    continue
                imin = skimage.io.imread(os.path.join(root, filename))
                imout = normalize_multi_channel(imin)
                skimage.io.imsave(os.path.join(local_output_folder, filename), imout)
        elif "masks" in root:
            for filename in files:
                shutil.copy(os.path.join(root, filename), local_output_folder)


    print 'DONE'