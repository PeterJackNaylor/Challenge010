# HistoNormalization2.py

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

# convert an image to 3 channels image. 
# will produce an error, if 2 channels are given. 
# will reduce the number of channels if more than 3 channels are given. 
def convert_to_3_channel_image(img):
    """
    converts an image to a 3-channel image. If the input image has more channels,
    the first 3 channels are taken. If the image contains only one channel, the channel
    is copied to all three channels. If the image contains 2 channels, an error is generated.

    arguments : img (numpy array with 2 or 3 dimensions)
    """
    nb_channels = img.shape[-1]
    if (nb_channels == 1):
        return(skimage.color.gray2rgb(img))
    if (nb_channels == 3):
        return(img)
    if (nb_channels > 3):
        return (img[:,:,:3])
    raise ValueError("image has %i channels. This number is not supported." % img.shape[-1])

def inv_channels(img):
    """
    inverts the channels of img, if the sum of squares on the left is larger than the 
    sum of squares to the right. The sum of squares is calculated with respect to the mode.

    arguments : img (numpy array with 2 or 3 dimensions)
    """
    output = img.copy()

    # if img is a single channel
    if (len(img.shape) < 3) or (img.shape[-1]==1):
        res = inv_channel(img)
        return(res)

    # if img has multiple channels
    for i in range(img.shape[-1]):
        output[:,:,i] = inv_channel(img[:,:,i])

    return output


def inv_channel(img):
    """
    inverts img, if the sum of squares on the left is larger than the 
    sum of squares to the right. The sum of squares is calculated with respect to the mode.
    This function works only for 2D single channel images. 

    arguments : img (numpy array with 2 dimensions)
    """
    im_histo = np.histogram(img, bins=32)
    histo_counts = im_histo[0]
    histo_values = im_histo[1]

    # get the background:
    index = np.argmax(histo_counts)
    background_value = .5 * (histo_values[index] + histo_values[index+1])

    N_left = np.float(np.sum(histo_counts[:index]))
    left_val = np.dot(histo_counts[:index].T / N_left, (histo_values[:index] - background_value)**2)
    N_right = np.float(np.sum(histo_counts[(index+1):]))
    right_val = np.dot(histo_counts[index:].T/N_right, (histo_values[(index+1):] - background_value)**2)

    if left_val > right_val:
        output = np.max(img) - img
    else:
        output = img

    return output


# gets the snr of an image by comparing the top 0.5% to the 10%
def get_snr_single_channel(imin):
    """
    gets the SNR of an image imin. Here, SNR is defined as the ratio of the expectation of 
    sum of squares of signal values and background mean and the expectation of sum of squares of 
    background values and background mean. Signal corresponds here to the .5% brightest pixels, noise 
    corresponds to the 10% darkest pixels. For this reason, inversion must be called prior to calling this
    function. 
    """
    imin_inv = inv_channel(imin)
    perc_thresholds = np.percentile(imin_inv, [10, 99.5])

    # test for empty image
    low_values = imin_inv[imin_inv <= perc_thresholds[0]]
    high_values = imin_inv[imin_inv >= perc_thresholds[1]]

    bg_mean = np.mean(low_values)
    # 0.5 corresponds to random fluctuations of value 1. 
    # this can be seen as a minimal amount of noise. 
    bg_noise = max( 0.5, np.std(low_values))

    # signal energy
    signal_energy = 1.0 / len(high_values) * np.sum((high_values - bg_mean)**2)

    snr = signal_energy / bg_noise

    return snr


def normalize_single_channel(imin, inv=True, maxval=255.0, minval=0.0):
    """
    normalizes a single channel grey level image. This is a percentile normalization.
    The 20 percentile is set as 0 value, the max in the image is set to the maximal possible
    value of the image type.

    Arguments: 
        imin        :   np.array, 2D
        inv         :   indicates whether inv_channel should be called
                        (Default = True)
        maxval      :   maxval to be expected for the image type (Default = 255)
        minval      :   minval to be expected for the image type (Default = 255)
    """
    snr = get_snr_single_channel(imin)
    if snr < 100.0: 
        return imin

    if inv:
        im_cp = inv_channel(imin)
    else:
        im_cp = imin.copy()
    perc = np.percentile(im_cp, [20, 100])

    im_cp = im_cp.astype(np.float)
    imin_norm = maxval * (im_cp - perc[0]) / (perc[1] - perc[0]) 
    imin_norm[imin_norm> maxval] = maxval
    imin_norm[imin_norm < minval] = minval 

    return imin_norm



def normalize_multi_channel(imin): 
    """
    normalizes a color or grey scale image by applying normalize_single_channel. 
    Grey scale images are first inverted according to the rule of inv_channel, i.e.
    if compared to the mode the MSE on the left is larger than the MSE on the right.
    In addition to this, there will be percentile histogram stretching with 20 and 99.5 
    as percentiles.

    Arguments: 
        imin       :   np.array, 2D or 3D
    """
    # convert to 3 channels
    colorin = convert_to_3_channel_image(imin)

    # grey or not grey
    is_grey_scale = (np.mean(colorin[:,:,0]) == np.mean(colorin[:,:,1]) == np.mean(colorin[:,:,2]))

    if is_grey_scale: 
        # if grey scale
        colorout = colorin.copy()
        for i in range(3):
            colorout[:,:,i] = normalize_single_channel(colorin[:,:,i])
        imout = colorout.astype(np.uint8)

    else:
        colorout = colorin.copy()
        for i in range(3):
            temp = np.max(colorin[:,:,i]) - colorin[:,:,i]  
            colorout[:,:,i] = normalize_single_channel(temp, maxval=255, minval=0, inv=False)
        imout = colorout.astype(np.uint8)

    return(imout)

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
        for filename in files:
            if filename[0] == ".":
                continue
            if filename.endswith('_mask.png'):
                shutil.copy(os.path.join(root, filename), local_output_folder)
            else:
                imin = skimage.io.imread(os.path.join(root, filename))
                imout = normalize_multi_channel(imin)
                skimage.io.imsave(os.path.join(local_output_folder, filename), imout)

    print 'DONE'


