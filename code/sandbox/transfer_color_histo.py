# import the necessary packages
import numpy as np
import skimage.color
from scipy.stats import skew
import pdb

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
    """
    Transfers the color distribution from the source to the target
    image using the mean and standard deviations of the L*a*b*
    color space.
    This implementation is (loosely) based on to the "Color Transfer
    between Images" paper by Reinhard et al., 2001.
    Parameters:
    -------
    source: NumPy array
        image in RGB color space (the source image)
    target: NumPy array
        image in RGB color space (the target image)
    Returns:
    -------
    transfer: NumPy array
        image (w, h, 3) NumPy array (uint8)
    """

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

