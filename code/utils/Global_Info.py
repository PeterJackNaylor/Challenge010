from glob import glob
from os import environ
from os.path import dirname, join
import pdb

def mask_dic(image_folder):
    dic = {}
    for el in image_folder:
        mask_path = dirname(el).replace("images", "masks")
        dic[el] = glob(join(mask_path, "*.png"))
    return dic

if 'WORKINGDIR' in environ.keys():
    print "Base directory is set to :", environ['WORKINGDIR']
    image_path = join(environ['WORKINGDIR'], "../dataset/stage1_train/*/images/*.png")
    image_test_path = join(environ['WORKINGDIR'], "../dataset/stage1_test/*/images/*.png")
else:
    print "Debuging nextflow folder"
    image_path = join("../../../..", "../dataset/stage1_train/*/images/*.png")
    image_test_path = join("../../../..", "../dataset/stage1_test/*/images/*.png")

image_files = glob(image_path)
image_test_files = glob(image_test_path)
masks_dic = mask_dic(image_files)

