import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import csv

import shutil

import pdb
NN_PRED_BASE_FOLDER = '/Users/twalter/data/Challenge_010/nn_out'

NN_VERSIONS = ['ContrastUNet', 'Dist', 'UNetHistogramTW1']
PRED_VIS_FOLDER = '/Users/twalter/data/Challenge_010/prediction_vis'
NN_ANALYSIS_PLOTS = '/Users/twalter/data/Challenge_010/nn_analysis'

def get_nn_pred_folder(nn_version='ContrastUNet'):
    temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, 'Validation')
    folders = filter(lambda x: x[0] != '.', os.listdir(temp_folder))
    res_folder = os.path.join(temp_folder, folders[0])
    return res_folder

def get_image_folder(img_name, nn_version='ContrastUNet'): 
    res_folder = os.path.join(get_nn_pred_folder(nn_version), img_name)
    return res_folder

def get_original_image(img_name, nn_version='ContrastUNet'):
    image_folder = get_image_folder(img_name, nn_version)
    image_name = os.path.join(image_folder, 'rgb.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_groundtruth_image(img_name, nn_version='ContrastUNet'):
    image_folder = get_image_folder(img_name, nn_version)
    image_name = os.path.join(image_folder, 'colored_bin.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_prediction_image(img_name, nn_version='ContrastUNet'):
    image_folder = get_image_folder(img_name, nn_version)
    image_name = os.path.join(image_folder, 'colored_pred.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_prob_image(img_name, nn_version='ContrastUNet'):
    image_folder = get_image_folder(img_name, nn_version)
    image_name = os.path.join(image_folder, 'output_DNN.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def read_nn_performance():
    info = {}
    for nn_version in NN_VERSIONS:
        nn_folder = get_nn_pred_folder(nn_version)
        filename = os.path.join(nn_folder, '__summary_per_image.csv')
        cols_to_retrieve = ['AJI', 'F1', 'path']

        fp = open(filename, 'r')

        content = csv.reader(fp)
        headers = content.next()
        
        header_indices = [headers.index(x) for x in cols_to_retrieve]
        
        for line in content:
            temp = dict(zip(cols_to_retrieve, [line[i] for i in header_indices]))
            img_name = os.path.basename(temp['path'])
            if not img_name in info:
                info[img_name] = {}
            info[img_name][nn_version] = dict(zip(cols_to_retrieve[:-1], [temp[x] for x in cols_to_retrieve[:-1]]))

        fp.close()

    # clean : remove all uncomplete data sets
    to_remove = []
    for img_name in info:
        if len(info[img_name]) < 3:
            to_remove.append(img_name)
    for img_name in to_remove:
        del(info[img_name])
    unclean = len(to_remove)    
    print '%i images were removed' % unclean            
    return info

def analyse_nn_performance(out_folder=None):
    if out_folder is None:
        out_folder = NN_ANALYSIS_PLOTS
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

    info = read_nn_performance()

    # plot F1 agains DSB
    # data preparation
    f1 = np.array([[np.float(info[x][nn]['F1']) for nn in info[x]] for x in info]).T
    aji = np.array([[np.float(info[x][nn]['AJI']) for nn in info[x]] for x in info]).T
    colors = ['red', 'green', 'blue']

    fig = plt.figure(figsize=(6,6))    
    for i in range(len(NN_VERSIONS)): 
        plt.scatter(f1[i], aji[i], c=colors[i], marker='.', s=5, label=NN_VERSIONS[i], edgecolor='')
    plt.xlabel('F1 score')
    plt.ylabel('AJI score')
    plt.title('F1 score vs. AJI score')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(out_folder, 'F1_AJI_all_nn.png'))
    plt.close('all')

    # plot dist against standard
    fig = plt.figure(figsize=(6,6))    
    plt.scatter(aji[0], aji[1], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('AJI score of UNet (contrast)')
    plt.ylabel('AJI score of DistNet')
    plt.title('Comaprison of dist net and UNet (AJI)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'Contrast_vs_Dist_AJI.png'))
    plt.close('all')

    fig = plt.figure(figsize=(6,6))    
    plt.scatter(f1[0], f1[1], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('F1 score of UNet (contrast)')
    plt.ylabel('F1 score of DistNet')
    plt.title('Comaprison of dist net and UNet (F1)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'Contrast_vs_Dist_F1.png'))
    plt.close('all')

    return

def display_prediction(img_name, out_folder=None):
    if out_folder is None:
        out_folder = PRED_VIS_FOLDER
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    im_orig = get_original_image(img_name)
    im_gt = get_groundtruth_image(img_name)

    fig, axs = plt.subplots(3, 3, tight_layout=True,
                            figsize=(15, 15))

    # original images (color)
    axs[0, 0].imshow(im_orig, norm=NoNorm())
    axs[0,0].axis('off')
    axs[1, 0].imshow(im_gt, norm=NoNorm())
    axs[1,0].axis('off')
    axs[2, 0].axis('off')
    #axs[0, 0].get_xaxis().set_visible(False)
    #axs[0, 0].get_yaxis().set_visible(False)

    for i in range(3):
        nn_version = NN_VERSIONS[i]
        prediction = get_prediction_image(img_name, nn_version)
        prob = get_prob_image(img_name, nn_version)
        axs[i, 1].imshow(prediction, cmap="gray", norm=NoNorm())
        axs[i, 1].axis('off')
        #axs[i, 1].set_ylabel(nn_version, size='large')
        
        axs[i, 2].imshow(prob, cmap="gray") #, norm=NoNorm())
        axs[i, 2].axis('off')

        #axs[i,2].annotate(nn_version, xy=(1, 0.5), xytext=(-axs[i,2].yaxis.labelpad - pad, 0),
        #                  xycoords=axs[i,2].yaxis.label, textcoords='offset points',
        #                  size='large', ha='right', va='center', rotation=90)
            
            #ax.set_ylabel(nn_version, size='large')

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    plt.savefig(os.path.join(out_folder, '%s_comparison.png' % img_name))
    plt.close('all')

    return 


