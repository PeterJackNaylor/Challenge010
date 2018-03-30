import uuid
from glob import glob
from utils.random_utils import CheckOrCreate
from os.path import join
from skimage.io import imread, imsave
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

