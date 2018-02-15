from utils.Global_Info import image_files, image_test_files, masks_dic
from skimage.io import imread
import numpy as np
import pdb
import os
from os.path import abspath, basename, join
from scipy.ndimage.morphology import morphological_gradient
from utils.random_utils import generate_wsl

def fuse(el, dic):
    """
    given a name and a dic, it will fuse all bin images in dic to one label image
    """
    img = imread(el)
    res = np.zeros_like(img[:,:,0], dtype="int")
    for k, mask_name in enumerate(dic[el]):
        bin_img = imread(mask_name)
        res[bin_img > 0] = k + 1
    return res

def meta_data_test(el, empty_dic):
    img = imread(el)

    if len(img.shape) == 3:
        x, y, z = img.shape[0:3]
    else:
        x, y = img.shape[0:3]
        z = 1

    empty_dic.loc[basename(el), "path_to_image"] = abspath(el)
    empty_dic.loc[basename(el), "type"] = img.dtype
    empty_dic.loc[basename(el), "shape_x"] = x
    empty_dic.loc[basename(el), "shape_y"] = y
    empty_dic.loc[basename(el), "shape_z"] = z

    mean = np.zeros(z, dtype=float)
    for i in range(z):
        mean[i] = np.mean(img[:,:,i])
        empty_dic.loc[basename(el), "channel_{}".format(i)] = mean[i]

    RGB = 1
    if mean[0] == mean[1]:
        if mean[1] == mean[2]:
            RGB = 0

    empty_dic.loc[basename(el), 'RGB'] = RGB
    WhiteBackGround = 0
    BlackBackGround = 0
    if RGB == 0:
        if mean[0] > 125:
            BackGround = "White"
            WhiteBackGround = 1
        else:
            BlackBackGround = 1
            BackGround = "Black"
    empty_dic.loc[basename(el), 'WhiteBackGround'] = WhiteBackGround
    empty_dic.loc[basename(el), 'BlackBackGround'] = BlackBackGround

def meta_data(el, mask_el, empty_dic):
    
    meta_data_test(el, empty_dic)
    empty_dic.loc[basename(el), "path_to_label"] = abspath(mask_el)

    img_mask = imread(mask_el)
    img_bin = img_mask.copy()
    img_bin[img_bin > 0] = 1

    if len(img_mask.shape) == 3:
        x, y, z = img_mask.shape[0:3]
    else:
        x, y = img_mask.shape[0:3]
        z = 1


    nuc_size = []
    for nuc_l in range(1, img_mask.max() + 1):
        nuc_size.append(len(np.where(img_mask == nuc_l)[0]))

    empty_dic.loc[basename(el), "number_of_nuclei"] = img_mask.max()
    empty_dic.loc[basename(el), "nuclei_avg_size"] = np.mean(nuc_size)
    empty_dic.loc[basename(el), "nuclei_max_size"] = np.max(nuc_size)
    empty_dic.loc[basename(el), "nuclei_med_size"] = np.median(nuc_size)
    empty_dic.loc[basename(el), "nuclei_min_size"] = np.min(nuc_size)
    empty_dic.loc[basename(el), "proportion_of_annotation"] = float(img_bin.sum()) / (x * y)


def split_into_domain(table):
    os.mkdir('domain_RGB')
    os.mkdir('domain_BlackOnWhite')
    os.mkdir('domain_WhiteOnBlack')

    def symlink_copy_to_folder(row):
        src = row["path_to_image"]
        dest = join('domain_{}', basename(src))
        if row['RGB']:
            dest = dest.format('RGB')
        elif row["WhiteBackGround"]:
            dest = dest.format('BlackOnWhite')
        else:
            dest = dest.format('WhiteOnBlack')
        os.symlink(src, dest)

    table.apply(lambda row: symlink_copy_to_folder(row), axis=1)    

def Contours(bin_image, contour_size=3):
    # Computes the contours
    grad = morphological_gradient(bin_image, size=(contour_size, contour_size))
    return grad

def Overlay(rgb, binary_image, black=True):
    res = imread(rgb)[:,:,0:3]
    mask = imread(binary_image)
    line = generate_wsl(mask)
    mask[mask > 0] = 1
    mask[line > 0] = 0
    mask = Contours(mask)
    if black: 
        res[mask > 0] = np.array([0, 0, 0])
    else:
        res[mask > 0] = np.array([255, 255, 255])
    return res
