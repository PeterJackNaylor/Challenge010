import uuid
from glob import glob
from utils.random_utils import CheckOrCreate
from os.path import join
from skimage.io import imread, imsave
from skimage.measure import label
import os
import numpy as np

def GRN():
    return str(uuid.uuid4()).replace("-", "")


TNBC = "../../../external_data/TNBC_NucleiSegmentation"
OUT = "../../../external_data/TNBC_Challenge"

PNG = glob(TNBC + "/Slide_*/*.png")

for png in PNG:
    mask = png.replace('/Slide_', '/GT_')
    name = GRN()
    CheckOrCreate(join(OUT, name))
    CheckOrCreate(join(OUT, name, 'images'))
    CheckOrCreate(join(OUT, name, 'masks'))
    os.symlink(png, join(OUT, name, 'images', name + ".png"))
    labels = label(imread(mask))
    for i in range(1, labels.max() + 1):
        single = np.zeros_like(labels, dtype='uint8')
        single[labels == i] = 255
        name_sing = GRN()
        imsave(join(OUT, name, 'masks', name_sing + ".png"), single)

