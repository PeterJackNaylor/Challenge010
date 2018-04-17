import os
import sys

import tensorflow as tf
ROOT_DIR = os.path.abspath("/Users/naylorpeter/Documents/Python/python3-packages/Mask_RCNN")
NUCLEUS_DIR = "/Users/naylorpeter/Documents/Python/python3-packages/Mask_RCNN/samples/nucleus"
LOGS_DIR = os.path.join("~/tmp", "model_rnn")
sys.path.append(ROOT_DIR) 
sys.path.append(NUCLEUS_DIR) 
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
import nucleus
import numpy as np
import pdb

from optparse import OptionParser
from skimage.io import imsave


def CheckOrCreate(path):
    if not os.path.isdir(path):
        os.makedirs(path)

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


parser = OptionParser()
#parser.add_option('--data', dest="data", type="str")
parser.add_option('--weights', dest="weights", type="str")
parser.add_option('--subset', dest="subset", type="str")
parser.add_option('--output', dest="output", type="str")
(options, args) = parser.parse_args()


def compute_mask(img_):
    x, y, z = img_.shape
    res = np.zeros(shape=(x, y))
    for i in range(img_.shape[2]):
        res[img_[:,:,i] > 0] = i+1
    return res
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# Only inference mode is supported right now
TEST_MODE = "inference"

config = nucleus.NucleusInferenceConfig()
config.display()
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference",
                              model_dir=LOGS_DIR,
                              config=config)
DATASET_DIR = "/Users/naylorpeter/Desktop/NucleiKaggle/dataset"
weights_path = "/Users/naylorpeter/tmp/model_rnn/ExternalData.h5"
weights_path = options.weights

print(weights_path)
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

dataset = nucleus.NucleusDataset()
dataset.load_nucleus(DATASET_DIR, options.subset)
dataset.prepare()

for image_id in dataset.image_ids:
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
    modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
                                           dataset.image_reference(image_id)))
    print("Original image shape: ", modellib.parse_image_meta(image_meta[np.newaxis,...])["original_image_shape"][0])

    # Run object detection
    results = model.detect_molded(np.expand_dims(image, 0), np.expand_dims(image_meta, 0), verbose=1)
    r = results[0]
    labeled_res = compute_mask(r['masks'])
    labeled_gt  = compute_mask(gt_mask)
    outfold = os.path.join(options.output, info['id'])
    CheckOrCreate(outfold)
    imsave(os.path.join(outfold, "rgb.png"), image.astype('uint8'))
    imsave(os.path.join(outfold, "label.png"), color_bin(labeled_gt.astype('uint8')))
    imsave(os.path.join(outfold, "pred.png"), color_bin(labeled_res.astype('uint8')))
