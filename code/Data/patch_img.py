from utils.Global_Info import image_files, image_test_files, masks_dic
from skimage.io import imread
from skimage import morphology as morp
from skimage import measure
import numpy as np
import pdb
import os
from os.path import abspath, basename, join
from scipy.ndimage.morphology import morphological_gradient
from utils.random_utils import generate_wsl
from sklearn import cluster
from sklearn import preprocessing
import pandas as pd

NCLUST = 6

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
    #pdb.set_trace()
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

def closenness(img_mask):
    labeled = measure.label(img_mask)
    cc = labeled.max()
    labeled[labeled > 0] = 1
    labeled = morp.dilation(labeled, morp.disk(3))
    labeled = measure.label(labeled)
    cc = cc - labeled.max()
    return cc


def meta_data(el, mask_el, empty_dic):
    
    meta_data_test(el, empty_dic)
    empty_dic.loc[basename(el), "path_to_label"] = abspath(mask_el)

    img = imread(el)
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
    #pdb.set_trace()
    for k in range(3):
        empty_dic.loc[basename(el), "mean_values_nuclei_channel_{}".format(k)] = img[:,:,k][img_bin == 1].mean()
        empty_dic.loc[basename(el), "std_values_nuclei_channel_{}".format(k)] = img[:,:,k][img_bin == 1].std()
        empty_dic.loc[basename(el), "mean_values_background_channel_{}".format(k)] = img[:,:,k][img_bin == 0].mean()
        empty_dic.loc[basename(el), "std_values_background_channel_{}".format(k)] = img[:,:,k][img_bin == 0].std()


    empty_dic.loc[basename(el), "closeness"] = closenness(img_mask)
    chan_0 = img[:,:,0].copy().astype('float')
    chan_0 = (chan_0 - chan_0.min()) / (chan_0.max() - chan_0.min())
    noise =  chan_0 - morp.opening(chan_0)
    noise_dect_nuc = np.mean(noise[img_bin == 1])
    noise_dect_back = np.mean(noise[img_bin == 0])
    empty_dic.loc[basename(el), "noise_dect_nuc"] = noise_dect_nuc
    empty_dic.loc[basename(el), "noise_dect_back"] = noise_dect_back


def split_into_domain(table):
    for i in range(NCLUST):
        os.mkdir('domain_RGB_group_{}'.format(i))
    for i in range(NCLUST + 4):
        os.mkdir('domain_BlackBackGround_group_{}'.format(i))
    os.mkdir('domain_WhiteBackGround')

    def symlink_copy_to_folder(row):
        src = row["path_to_image"]
        dest = join('domain_{}', basename(src))
        if row['RGB']:
            val = int(row["background_RGB"])
            dest = dest.format('RGB_group_{}'.format(val))
        elif row["WhiteBackGround"]:
            dest = dest.format('WhiteBackGround')
        else:
            val = int(row["background_BBG"])
            dest = dest.format('BlackBackGround_group_{}'.format(val))
            
        os.symlink(src, dest)

    table.apply(lambda row: symlink_copy_to_folder(row), axis=1)    

def split_into_domain_test(table):
    os.mkdir('domain_RGB')
    os.mkdir('domain_BlackBackGround')
    os.mkdir('domain_WhiteBackGround')

    def symlink_copy_to_folder(row):
        src = row["path_to_image"]
        dest = join('domain_{}', basename(src))
        if row['RGB']:
            dest = dest.format('RGB')
        elif row["WhiteBackGround"]:
            dest = dest.format('WhiteBackGround')
        else:
            dest = dest.format('BlackBackGround')

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

def Overlay_with_pred(rgb, pred_image, black=True):
    res = imread(rgb)[:,:,0:3]
    mask = pred_image
    line = generate_wsl(mask)
    mask[mask > 0] = 1
    mask[line > 0] = 0
    mask = Contours(mask)
    if black: 
        res[mask > 0] = np.array([0, 0, 0])
    else:
        res[mask > 0] = np.array([255, 255, 255])
    return res

def scale(table):
    x = table.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    return df

def unsupervised_groups(tab):
    """
    Divides blackbackground images in two groups and rgb ones
    """
    only_rgb = tab[tab["RGB"] == 1]
    only_bbg = tab[tab["BlackBackGround"] == 1]
    features_str = ["mean_values_nuclei_channel_{}".format(k) for k in range(0,3)] +  \
                        ["mean_values_background_channel_{}".format(k) for k in range(0,3)]  +\
                        ["std_values_nuclei_channel_{}".format(k) for k in range(0,3)]  +\
                        ["std_values_background_channel_{}".format(k) for k in range(0,3)]  +\
                        ["nuclei_avg_size", "number_of_nuclei"]
    features_back = ["mean_values_background_channel_{}".format(k) for k in range(0,3)] +\
                    ["std_values_background_channel_{}".format(k) for k in range(0,3)]
    features_nuc = ["mean_values_nuclei_channel_{}".format(0)] + \
                    ["std_values_nuclei_channel_{}".format(0)] + \
                    ["nuclei_avg_size", "noise_dect_nuc", "noise_dect_back"] +\
                    ["shape_x", "shape_y"]
                    #["nuclei_avg_size", "number_of_nuclei", ""]

    feat_ = only_rgb[features_back]
    feat_bbg = only_bbg[features_nuc]

    feat_ = scale(feat_)
    feat_bbg = scale(feat_bbg)


    #model = cluster.KMeans(n_clusters=NCLUST)
    model = cluster.AgglomerativeClustering(n_clusters=NCLUST)
    model_3 = cluster.AgglomerativeClustering(n_clusters=NCLUST + 4)

    tab.loc[tab["RGB"] == 1 , "background_RGB"] = model.fit_predict(feat_)
    tab.loc[tab["BlackBackGround"] == 1 , "background_BBG"] = model_3.fit_predict(feat_bbg)

    return tab