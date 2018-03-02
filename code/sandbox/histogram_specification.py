# test different histogram specification techniques
import transfer_color_histo
reload(transfer_color_histo)
import os, fnmatch
import skimage.io
import numpy as np
import pdb

# a reference image needs to be chosen
reference_img = '1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5.png'

def find_reference_image(in_folder, img_name = reference_img):
    matches = []
    for root, dirnames, filenames in os.walk(in_folder):
        for filename in fnmatch.filter(filenames, img_name):
            matches.append(os.path.join(root, filename))
    return matches

# applies color transfer as provided by the package color_transfer
# simply to all images.
def brute_force(in_folder, out_folder):
    reference_img_names = find_reference_image(in_folder)
    if len(reference_img_names) == 0:
        raise ValueError("reference image not found")
    reference_img = skimage.io.imread(reference_img_names[0])
    if reference_img.shape[-1] > 3:
        reference_img = reference_img[:,:,:3]
    reference_img = 255.0 * (reference_img - np.min(reference_img)) / (np.max(reference_img) - np.min(reference_img))
    reference_img = reference_img.astype(np.uint8)
    skimage.io.imsave(os.path.join(out_folder, '000_reference.png'), reference_img)

    for folder in os.listdir(in_folder):
        full_folder = os.path.join(in_folder, folder, 'images')
        filenames = os.listdir(full_folder)
        if len(filenames) == 0:
            print 'folder %s is empty' % full_folder
            continue
        if len(filenames) > 1:
            print 'several files found in folder %s' % full_folder
            continue
        imin = skimage.io.imread(os.path.join(full_folder, filenames[0]))
        skimage.io.imsave(os.path.join(out_folder, filenames[0]), imin)

        imflip = transfer_color_histo.flip_channels(imin)
        imout = transfer_color_histo.transfer_color(reference_img, imflip)
        skimage.io.imsave(os.path.join(out_folder, filenames[0].replace('.png', '_flip.png')), imflip)
        skimage.io.imsave(os.path.join(out_folder, filenames[0].replace('.png', '_normalized.png')), imout)

    return

