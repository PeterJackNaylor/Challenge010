import uuid
from glob import glob
from utils.random_utils import CheckOrCreate
from os.path import join
from skimage.io import imread, imsave
from skimage.measure import label
import os
import numpy as np
from optparse import OptionParser
import pdb
def GRN():
    return str(uuid.uuid4()).replace("-", "")

parser = OptionParser()
parser.add_option('--input', dest="input", type="str")
(options, args) = parser.parse_args()

TNBC = options.input
OUT = "."

PNG = glob(TNBC + "/data/images/dna-images/gnf/*.png") + glob(TNBC + "/data/images/dna-images/ic100/*.png")

def FindMask(name):
    return name.replace('images/dna-images', 'preprocessed-data')

for png in PNG:
    mask = FindMask(png)
    name = GRN()
    CheckOrCreate(join(OUT, name))
    CheckOrCreate(join(OUT, name, 'images'))
    CheckOrCreate(join(OUT, name, 'masks'))
    os.symlink(os.path.abspath(png), join(OUT, name, 'images', name + ".png"))
    labels = label(imread(mask))
    for i in range(1, labels.max() + 1):
        single = np.zeros_like(labels, dtype='uint8')
        single[labels == i] = 255
        name_sing = GRN()
        imsave(join(OUT, name, 'masks', name_sing + ".png"), single)

