# test different histogram specification techniques
import transfer_color_histo
reload(transfer_color_histo)
import os, fnmatch
import skimage.io
import numpy as np
import pdb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
from transfer_color_histo import convert_to_3_channel_image
from sklearn.neighbors import KernelDensity
import time


# a reference image needs to be chosen
#reference_img = '1b518cd2ea84a389c267662840f3d902d0129fab27696215db2488de6d4316c5.png'
reference_img = '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png'

def find_reference_image(in_folder, img_name = reference_img):
    matches = []
    for root, dirnames, filenames in os.walk(in_folder):
        for filename in fnmatch.filter(filenames, img_name):
            matches.append(os.path.join(root, filename))
    return matches

def open_reference_image(in_folder, img_name = reference_img):
    
    reference_img_names = find_reference_image(in_folder, img_name)
    
    if len(reference_img_names) == 0:
        raise ValueError("reference image not found")
    reference_img = skimage.io.imread(reference_img_names[0])
    if reference_img.shape[-1] > 3:
        reference_img = reference_img[:,:,:3]
    return reference_img 

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

IMAGE_LIST = [
'7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80.png',
'adc315bd40d699fd4e4effbcce81cd7162851007f485d754ad3b0472f73a86df.png',
'b1eb0123fe2d8c825694b193efb7b923d95effac9558ee4eaf3116374c2c94fe.png',
'f9ea1a1159c33f39bbe5f18bb278d961188b40508277eab7c0b4b91219b37b5d.png',
'2a2032c4ed78f3fc64de7e5efd0bec26a81680b07404eaa54a1744b7ab3f8365.png',
'bbce7ebc40323a0eff6574d0c3842f50f907f55fbfb46c777f0ed9a49e98ff9b.png',
'58406ed8ef944831c413c3424dc2b07e59aef13eb1ff16acbb3402b38b5de0bd.png',
'2869fad54664677e81bacbf00c2256e89a7b90b69d9688c9342e2c736ff5421c.png',
'b6c9b58de0388891221b8f7a83cbf0b8f8379b51b5c9a127bf43a4fc49f1cc48.png',
'b0defa611b75645c0283464ee4163917bad382d335b61e8509f065bf371fa15f.png',
'1609b1b8480ee52652a644403b3f7d5511410a016750aa3b9a4c8ddb3e893e8e.png',
'50a7ea80dd73232a17f98b5c83f62ec89989e892fe25b79b36f99b3872a7d182.png',
'89be66f88612aae541f5843abcd9c015832b5d6c54a28103b3019f7f38df8a6d.png',
'e5a7b8a9924b26b3abf039255a8a3bb00258f4966f68ff3349560b4350af9367.png',
              '0000.png',
              '866a8cba7bfe1ea73e383d6cf492e53752579140c8b833bb56839a55bf79d855.png',
              '9ebcfaf2322932d464f15b5662cae4d669b2d785b8299556d73fffcae8365d32.png',
              '7b38c9173ebe69b4c6ba7e703c0c27f39305d9b2910f46405993d2ea7a963b80.png',
              '93c5638e7e6433b5c9cc87c152bcbe28873d2f9d6a392cca0642520807542a77.png',
              '220b37f4ca7cab486d2b71cd87a46ee7411a5aa142799d96ed98015ab5ba538a.png',
              '84eeec681987753029eb83ea5f3ff7e8b5697783cdb2035f2882d40c9a3f1029.png',
              'e5aeb5b3577abbebe8982b5dd7d22c4257250ad3000661a42f38bf9248d291fd.png',
              '317832f90f02c5e916b2ac0f3bcb8da9928d8e400b747b2c68e544e56adacf6b.png',
            'f01a9742c43a69f087700a43893f713878e537bae8e44f76b957f09519601ad6.png',
              '04acab7636c4cf61d288a5962f15fa456b7bde31a021e5deedfbf51288e4001e.png',
              '1a11552569160f0b1ea10bedbd628ce6c14f29edec5092034c2309c556df833e.png',
              '00ae65c1c6631ae6f2be1a449902976e6eb8483bf6b0740d00530220832c6d3e.png',
              'af576e8ec3a8d0b57eb6a311299e9e4fd2047970d3dd9d6f52e54ea6a91109da.png',
              'a7f6194ddbeaefb1da571226a97785d09ccafc5893ce3c77078d2040bccfcb77.png',
              'a31deaf0ac279d5f34fb2eca80cc2abce6ef30bd64e7aca40efe4b2ba8e9ad3d.png',
              '52a6b8ae4c8e0a8a07a31b8e3f401d8811bf1942969c198e51dfcbd98520aa60.png',
              '13f2bec0a24c70345372febb14c4352877b1b6c1b01896246048e83c345c0914.png',
              '7798ca1ddb3133563e290c36228bc8f8f3c9f224e096f442ef0653856662d121.png',
              '8f27ebc74164eddfe989a98a754dcf5a9c85ef599a1321de24bcf097df1814ca.png', 
              '3bfa8b3b01fd24a28477f103063d17368a7398b27331e020f3a0ef59bf68c940.png', 
              '57bd029b19c1b382bef9db3ac14f13ea85e36a6053b92e46caedee95c05847ab.png',
              '0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466.png',
              '1d9eacb3161f1e2b45550389ecf7c535c7199c6b44b1c6a46303f7b965e508f1.png',
              '9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab.png',
              '17b9bf4356db24967c4677b8376ac38f826de73a88b93a8d73a8b452e399cdff.png',
              '259b35151d4a7a5ffdd7ab7f171b142db8cfe40beeee67277fac6adca4d042c4.png',
              '472b1c5ff988dadc209faea92499bc07f305208dbda29d16262b3d543ac91c71.png',
              ]


def get_background_color(img):
    im_histo = np.histogram(img, n_bins=32)

    # get the background:
    index = np.argmax(histo_counts)
    background_value = .5 * (histo_values[index] + histo_values[index+1])

    return background_value

def inv_channels(img):
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

def get_snr_single_channel(imin):
    #start_time = time.time()
    imin_inv = inv_channel(imin)
    perc_thresholds = np.percentile(imin_inv, [10, 99.5])

    # test for empty image
    low_values = imin_inv[imin_inv <= perc_thresholds[0]]
    high_values = imin_inv[imin_inv >= perc_thresholds[1]]

    bg_mean = np.mean(low_values)
    # 0.5 corresponds to random fluctuations of value 1. 
    # this can be seen as a minimal amount of noise. 
    bg_noise = max( 0.5, np.std(low_values))

#    print perc_thresholds
#    print bg_mean, bg_noise
#    print len(low_values), len(high_values)
#    if len(low_values) == 22522:
#        pdb.set_trace()

    signal_energy = 1.0 / len(high_values) * np.sum((high_values - bg_mean)**2)

    snr = signal_energy / bg_noise
    # test for empty image

    # im_histo = np.histogram(img, n_bins=128, density=True)
    #print 'image dimensions: [%i, %i] \t time elapsed: %.2f sec' % (imin_inv.shape[0], imin_inv.shape[1],(time.time() - start_time))
    return snr

def normalize_single_channel(imin, inv=True, maxval=255.0, minval=0.0):
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
    #res = imin_norm.astype(np.uint8)

    return imin_norm

def col_transfer(source, target):
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

    source_lab = skimage.color.rgb2lab(source)
    target_lab = skimage.color.rgb2lab(target)

    s_mean = np.mean(source_lab, axis=(0, 1))
    s_std = np.std(source_lab, axis=(0, 1))
    t_mean = np.mean(target_lab, axis=(0, 1))
    t_std = np.std(target_lab, axis=(0, 1))

    for i in range(1,3):
        target_lab[:,:,i] = (target_lab[:,:,i] - t_mean[i]) * s_std[i] / t_std[i] + s_mean[i]

    output = 255 * skimage.color.lab2rgb(target_lab)
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return output

def TEMP____col_transfer(source, target, channels=None):
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
    if channels is None:
        channels = (1, 1, 1)

    source_lab = skimage.color.rgb2lab(source)
    target_lab = skimage.color.rgb2lab(target)

    perc = np.percentile(source_lab, [20, 95.5])

    s_mean = np.mean(source_lab, axis=(0, 1))
    s_std = np.std(source_lab, axis=(0, 1))
    t_mean = np.mean(target_lab, axis=(0, 1))
    t_std = np.std(target_lab, axis=(0, 1))

    for i in range(1,3):
        target_lab[:,:,i] = (target_lab[:,:,i] - t_mean[i]) * s_std[i] / t_std[i] + s_mean[i]

    output = 255 * skimage.color.lab2rgb(target_lab)
    output = np.clip(output, 0, 255)
    output = output.astype(np.uint8)

    return output

def normalize_multi_channel_old(imin, color_option=None, in_folder=None, ref_img_name=reference_img): 

    if color_option is None:
        color_option = ('Lab', (1, 0, 0))

    if (len(imin.shape) < 3) or (imin.shape[-1] == 1):
        imout = normalize_single_channel(imin)
        return(imout.astype(np.uint8))

    nb_channels = imin.shape[-1]
    is_grey_scale = (np.mean(imin[:,:,0]) == np.mean(imin[:,:,1]) == np.mean(imin[:,:,2]))
    if nb_channels == 2:
        raise ValueError("only 2 channels ... this format is not supported by the color normalization.")
    if nb_channels > 3:
        imin_safe = imin[:,:,:3]
        nb_channels = 3
    else:
        imin_safe = imin

    if is_grey_scale: 
        # if grey scale
        imout = imin_safe.copy()
        channel_norm = normalize_single_channel(imin_safe[:,:,0])
        for i in range(nb_channels):
            imout[:,:,i] = channel_norm.astype(np.uint8)

    else:
        if color_option[0] == 'Lab':
            colorin = skimage.color.rgb2lab(imin_safe)
            maxval = 1.0
            minval = 0.0
        if color_option[0] == 'RGB':
            colorin = imin_safe
            maxval = 255
            minval = 0
        if color_option[0] == 'HSV':
            colorin = skimage.color.rgb2hsv(imin_safe)
            maxval = 1.0
            minval = 0.0
        if color_option[0] == 'ref':
            ref_img = open_reference_image(in_folder, ref_img_name)
            colorin = col_transfer(ref_img, imin_safe)
            maxval = 1.0
            minval = 0.0

        colorout = colorin.copy()
        for i, do_channel in enumerate(color_option[1]):
            if (do_channel):
                colorout[:,:,i] = normalize_single_channel(colorin[:,:,i], maxval=maxval, minval=minval)

        if color_option[0] == 'Lab':
            imout = 255*skimage.color.lab2rgb(colorout)
            imout = imout.astype(np.uint8)
        if color_option[0] == 'RGB':
            imout = colorout.astype(np.uint8)
        if color_option[0] == 'HSV':
            imout = 255.0 * skimage.color.hsv2rgb(colorout)
            imout = imout.astype(np.uint8)
        if color_option[0] == 'ref':
            imout = colorout.astype(np.uint8)
            #imout = 255*skimage.color.lab2rgb(colorout)
            #imout = imout.astype(np.uint8)


    return(imout)


def normalize_multi_channel(imin): 

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

def plot_histo(in_folder, out_folder, selected_images=None,
               kernel_density=False): 
    if selected_images is None:
        selected_images = IMAGE_LIST
    n_bins = 32
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    for folder in os.listdir(in_folder):
        if folder[0] == '.':
            continue
        full_folder = os.path.join(in_folder, folder, 'images')
        filenames = os.listdir(full_folder)
        if len(filenames) == 0:
            print 'folder %s is empty' % full_folder
            continue
        if len(filenames) > 1:
            print 'several files found in folder %s' % full_folder
            continue
        if len(selected_images) > 0 and filenames[0] not in selected_images:
            continue

        report_str = filenames[0]
        imin = skimage.io.imread(os.path.join(full_folder, filenames[0]))
        imnorm = normalize_multi_channel(imin)
        #imnorm = normalize_multi_channel(imin, ('Lab', (1, 0, 0) ))
        #imnorm = normalize_multi_channel(imin, ('HSV', (0, 1, 1) ))
        #imnorm = normalize_multi_channel(imin, ('ref', (0, 0, 0) ), in_folder=in_folder)
#        imnorm = imin.copy()
#        for i in range(3):
#            imnorm[:,:,i] = normalize_single_channel(imin[:,:,i])

        #if np.mean(imin[:,:,0] != imin[:,:,1]):
        #    imin = skimage.color.rgb2lab(convert_to_3_channel_image(imin))
        fig, axs = plt.subplots(4, 4, tight_layout=True,
                                figsize=(12, 12))

        # original images (color)
        axs[0, 0].imshow(imin, norm=NoNorm())
        axs[1, 0].imshow(imnorm, norm=NoNorm())

        for i in range(1,4):
            channel_index = i-1
            #snr = get_snr_single_channel(imin[:,:,i])
            #report_str += ' %f' % snr
            # show unnormalized image
            axs[0, i].imshow(imin[:,:,channel_index], cmap='gray', norm=NoNorm())
            axs[0, i].get_xaxis().set_visible(False)
            axs[0, i].get_yaxis().set_visible(False)

            axs[1, i].imshow(imnorm[:,:,channel_index], cmap='gray', norm=NoNorm())
            axs[1, i].get_xaxis().set_visible(False)
            axs[1, i].get_yaxis().set_visible(False)

            # histogram (unnormalized)
            hist_obj = axs[2, i].hist(imin[:,:,channel_index].ravel(), bins=n_bins, log=False,
                                      normed=True)
            
            if kernel_density:
                # kernel density estimation
                X = imin[:,:,channel_index].ravel().reshape(-1, 1)

                kde = KernelDensity(kernel='gaussian', bandwidth=4.0).fit(X)
                log_dens = kde.score_samples(hist_obj[1].reshape(-1, 1))
                axs[2,i].plot(hist_obj[1], np.exp(log_dens), '-', color='r')

            # plot axis
            axs[2,i].set_ylim(0, np.percentile(hist_obj[0], 95))

            # histogram (normalized image)
            hist_obj = axs[3, i].hist(imnorm[:,:,channel_index].ravel(), bins=n_bins, log=False,
                                      normed=True)
            
            if kernel_density:
                # kernel density estimation
                X = imnorm[:,:,i].ravel().reshape(-1, 1)

                kde = KernelDensity(kernel='gaussian', bandwidth=4.0).fit(X)
                log_dens = kde.score_samples(hist_obj[1].reshape(-1, 1))
                axs[2,i].plot(hist_obj[1], np.exp(log_dens), '-', color='r')

            # plot axis
            axs[3,i].set_ylim(0, np.percentile(hist_obj[0], 95))

        #print report_str 
        plt.savefig(os.path.join(out_folder, filenames[0].replace('.png', '_histo.pdf')))
        plt.close('all')
    return


# def hist_match(source, template):
#     """
#     Adjust the pixel values of a grayscale image such that its histogram
#     matches that of a target image

#     Arguments:
#     -----------
#         source: np.ndarray
#             Image to transform; the histogram is computed over the flattened
#             array
#         template: np.ndarray
#             Template image; can have different dimensions to source
#     Returns:
#     -----------
#         matched: np.ndarray
#             The transformed output image
#     """

#     oldshape = source.shape
#     source = source.ravel()
#     template = template.ravel()

#     # get the set of unique pixel values and their corresponding indices and
#     # counts
#     s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
#                                             return_counts=True)
#     t_values, t_counts = np.unique(template, return_counts=True)

#     # take the cumsum of the counts and normalize by the number of pixels to
#     # get the empirical cumulative distribution functions for the source and
#     # template images (maps pixel value --> quantile)
#     s_quantiles = np.cumsum(s_counts).astype(np.float64)
#     s_quantiles /= s_quantiles[-1]
#     t_quantiles = np.cumsum(t_counts).astype(np.float64)
#     t_quantiles /= t_quantiles[-1]

#     # interpolate linearly to find the pixel values in the template image
#     # that correspond most closely to the quantiles in the source image
#     interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

#     return interp_t_values[bin_idx].reshape(oldshape)


