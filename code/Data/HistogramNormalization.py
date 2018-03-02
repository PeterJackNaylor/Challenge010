from optparse import OptionParser
import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
from utils.random_utils import CheckOrCreate
from scipy.stats import skew
import shutil

import pdb

# a reference image needs to be chosen
reference_image_name = '1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5.png'
def find_reference_image(in_folder, img_name = reference_image_name):
    matches = []
    for root, dirnames, filenames in os.walk(in_folder):
        for filename in fnmatch.filter(filenames, img_name):
            matches.append(os.path.join(root, filename))
    return matches

def get_reference_image(in_folder, filename=reference_image_name): 
    reference_img_names = find_reference_image(in_folder, img_name=filename)
    if len(reference_img_names) == 0:
        raise ValueError("reference image not found")
    reference_img = skimage.io.imread(reference_img_names[0])
    if reference_img.shape[-1] > 3:
        reference_img = reference_img[:,:,:3]
    reference_img = 255.0 * (reference_img - np.min(reference_img)) / (np.max(reference_img) - np.min(reference_img))
    reference_img = reference_img.astype(np.uint8)
    return reference_img 

def flip_channels(img):
    output = img.copy()
    for i in range(img.shape[-1]):
        s = skew(img[:,:,i].flatten())
        if s < 0:
            output[:,:,i] = np.max(img[:,:,i]) - img[:,:,i]
    return output

def convert_to_3_channel_image(img):
    nb_channels = img.shape[-1]
    if nb_channels == 1:
        return skimage.color.gray2rgb(img)
    if nb_channels == 3:
        return img
    if nb_channels > 3:
        return img[:,:,:3]
    raise ValueError("image has %i channels. This number is not supported." % img.shape[-1])

def transfer_color(source, target):
    source_lab = skimage.color.rgb2lab(convert_to_3_channel_image(source))
    target_lab = skimage.color.rgb2lab(convert_to_3_channel_image(target))

    s_mean = np.mean(source_lab, axis=(0, 1))
    s_std = np.std(source_lab, axis=(0, 1))
    t_mean = np.mean(target_lab, axis=(0, 1))
    t_std = np.std(target_lab, axis=(0, 1))

    for i in range(3):
        target_lab[:,:,i] = (target_lab[:,:,i] - t_mean[i]) * s_std[i] / t_std[i] + s_mean[i]

    output = skimage.color.lab2rgb(target_lab)
    output = np.clip(output, 0, 255)

    return output

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

    # get reference image
    reference_img = get_reference_image(input_folder)

    for root, dirs, files in os.walk(input_folder):        
        local_output_folder = root.replace(input_folder, output_folder)
        CheckOrCreate(local_output_folder)
        for filename in files:
            if filename.endswith('_mask.png'):
                shutil.copy(os.path.join(root, filename), local_output_folder)
            else:
                imin = skimage.io.imread(os.path.join(root, filename))
                imflip = flip_channels(imin)
                imout = transfer_color(reference_img, imflip)
                skimage.io.imsave(os.path.join(local_output_folder, filename), imout)

    print 'DONE'

