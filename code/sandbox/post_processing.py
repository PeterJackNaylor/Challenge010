import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
from skimage.measure import label, regionprops

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import csv

import shutil
from matplotlib.ticker import NullFormatter

import skimage.morphology 

import pdb
import sys

sys.path.append('/Users/twalter/pycharm/Challenge010/code')

from Data.HistogramNormalization2 import normalize_multi_channel

NN_PRED_BASE_FOLDER = '/Users/twalter/data/Challenge_010/nn_out'

NN_VERSIONS = ['ContrastUNet', 'Dist', 'UNetHistogramTW1']
PRED_VIS_FOLDER = '/Users/twalter/data/Challenge_010/prediction_vis'
NN_ANALYSIS_PLOTS = '/Users/twalter/data/Challenge_010/nn_analysis'

def get_all_images():
    nn_version='ContrastUNet'
    temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, 'Validation')
    folders = filter(lambda x: x[0] != '.' and os.path.isdir(os.path.join(temp_folder, x)), os.listdir(temp_folder))
    sample_folder = os.path.join(temp_folder, folders[0])
    image_folders = filter(lambda x: x[0] != '.' and os.path.isdir(os.path.join(sample_folder, x)), os.listdir(sample_folder))
    return image_folders

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

def get_image_shape_features(label_img):
    
    props = skimage.measure.regionprops(label_img)
    area = np.array([obj.area for obj in props])
    
    eigenvalue_1 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) + 
                             .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                             for obj in props])
    eigenvalue_2 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) - 
                             .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                             for obj in props])
    ellipse_area = np.pi * 4.0 * np.sqrt(eigenvalue_1 * eigenvalue_2) / area.astype(np.float)

    ellipse_feature = np.abs(area.astype(np.float) - ellipse_area) / np.clip(ellipse_area, a_min=1.0, a_max=None)

    bbox = [obj.bbox for obj in props]
    indices = np.where( (np.array([min((x[2] - x[0]), (x[3] - x[1])) for x in bbox]) > 2) * (area > 100))

    res = {
            'area': area[indices],
            'ellipse_feature': ellipse_feature[indices],
            'major_axis_length': np.array([obj.major_axis_length for obj in props])[indices],
            'bbox': np.array(bbox)[indices],
    }
    return res

def get_shape_features(image_list=None):
    temp_folder = os.path.join(NN_ANALYSIS_PLOTS, 'bad_ex')
    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    if image_list is None:
        image_list = get_all_images()
    res = {}
    for image_name in image_list:
        ground_truth_image = get_groundtruth_image(image_name)
        bin_img = ground_truth_image[:,:,0]>0
        label_img = label(bin_img, neighbors=4)
        res[image_name] = get_image_shape_features(label_img)
        indices = np.where(res[image_name]['ellipse_feature'] > 1.5)
        #pdb.set_trace()
        if len(indices[0]) > 0:
            print image_name, res[image_name]['area'][indices], ' vs ', res[image_name]['ellipse_feature'][indices]

            for min_row, min_col, max_row, max_col in res[image_name]['bbox'][indices]:
                print '\t', min_row, max_row, min_col, max_col
                small_img = 255*bin_img[min_row:max_row, min_col:max_col]
                skimage.io.imsave(os.path.join(temp_folder, '%s_%i_%i_%i_%i.png' % (image_name, min_row, min_col, max_row, max_col) ),
                                  small_img)

    return res

def preprocessing_ellipse_tophat(img, ellipse_lambda=0.1):
    if len(img.shape) > 2:
        imin = img[:,:,0].copy()
    else: 
        imin = img.copy()
    total_area = imin.shape[0] * imin.shape[1]

    im_ellipse_open = skimage.morphology.ellipse_filter(imin, ellipse_lambda,
                                                        area_low_thresh=100, 
                                                        area_high_thresh=8000,
                                                        method='cut_first')
    im_ellipse_tophat = imin - im_ellipse_open

    return im_ellipse_tophat

def preprocessing_ellipse_filter(img, ellipse_lambda=0.9):
    if len(img.shape) > 2:
        imin = img[:,:,0].copy()
    else: 
        imin = img.copy()
    total_area = imin.shape[0] * imin.shape[1]

    im_ellipse_open = skimage.morphology.ellipse_filter(imin, ellipse_lambda,
                                                        area_low_thresh=100, 
                                                        area_high_thresh=8000,
                                                        method='direct')

    return im_ellipse_open

def run_preprocessing_ellipse_tophat(image_list=None, out_folder=None, ellipse_lambda=0.1,
                                     ellipse_filter_lambda=0.9):

    if image_list is None:
        image_list = get_all_images()

    if out_folder is None:
        out_folder = '/Users/twalter/data/Challenge_010/ellipse_filter_out1'
    
    img_out_folder = os.path.join(out_folder, 'ellipse_tophat')
    comp_out_folder = os.path.join(out_folder, 'comparison')
    for folder in [img_out_folder, comp_out_folder]:
        if not os.path.isdir(folder):
            os.makedirs(folder)

    for image_name in image_list[:20]:
        # processing
        orig_image = get_original_image(image_name)
        imin = normalize_multi_channel(orig_image)
        out_name = os.path.join(img_out_folder, '%s_original.png' % image_name)
        skimage.io.imsave(out_name, imin)

        ellipse_tophat = preprocessing_ellipse_tophat(imin, ellipse_lambda)
        ellipse_tophat = 255.0 * ellipse_tophat.astype(np.float) / np.max(ellipse_tophat)
        ellipse_tophat[ellipse_tophat > 255.0] = 255.0
        ellipse_tophat = ellipse_tophat.astype(np.uint8)

        ellipse_filter = preprocessing_ellipse_filter(imin, ellipse_filter_lambda)
        ellipse_filter = 255.0 * ellipse_filter.astype(np.float) / np.max(ellipse_filter)
        ellipse_filter[ellipse_filter > 255.0] = 255.0
        ellipse_filter = ellipse_filter.astype(np.uint8)

        # save image
        out_name = os.path.join(img_out_folder, '%s_ellipse_tophat.png' % image_name)
        skimage.io.imsave(out_name, ellipse_tophat)

        out_name = os.path.join(img_out_folder, '%s_ellipse_filter.png' % image_name)
        skimage.io.imsave(out_name, ellipse_filter)

        # save comparison
        fig, axs = plt.subplots(1, 3, tight_layout=True,
                            figsize=(24, 8))

        # original images (color)
        implot = 255 * imin[:,:,0].astype(np.float) / imin.max()
        implot[implot > 255.0] = 255.0
        implot = implot.astype(np.uint8)
        axs[0].imshow(implot, cmap='gray', norm=NoNorm())
        axs[0].axis('off')
        axs[1].imshow(ellipse_tophat, cmap='gray', norm=NoNorm())
        axs[1].axis('off')
        axs[2].imshow(ellipse_filter, cmap='gray', norm=NoNorm())
        axs[2].axis('off')

        filename = '%s_ellipse_tophat_comparison.png' % image_name
        plt.savefig(os.path.join(comp_out_folder, filename))
        plt.close('all')
    return

def plot_groundtruth_features(feature_dict, feature1, feature2): 

    filename = os.path.join(NN_ANALYSIS_PLOTS, 'ground_truth_scatter_%s_%s.pdf' % (feature1, feature2))
    x = np.concatenate([np.array(feature_dict[image_name][feature1]) for image_name in feature_dict])
    y = np.concatenate([np.array(feature_dict[image_name][feature2]) for image_name in feature_dict])
    
    for vec, feature_name in zip([x, y], [feature1, feature2]):
        perc = np.percentile(vec, [90, 95, 99])
        print 'feature distribution of %s' % feature_name
        print 'mean: %.3f\tstd: %.3f' % (np.mean(vec), np.std(vec))
        print 'median: %.3f\t90-perc:%.3f\t95-perc:%.3f\t99-perc:%.3f' % (np.median(vec), perc[0], perc[1], perc[2])
        print 
    nb_bins = 100
    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    # start with a rectangular Figure
    plt.figure(1, figsize=(8, 8))

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.scatter(x, y, marker='.', s=20)
    axScatter.grid(linestyle=':')

    # now determine nice limits by hand:
    #binwidth = 0.25
    #xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    #lim = (int(xymax/binwidth) + 1) * binwidth

    #axScatter.set_xlim((-lim, lim))
    #axScatter.set_ylim((-lim, lim))

    #bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, bins=nb_bins)
    axHistx.grid(linestyle=':')
    axHisty.hist(y, bins=nb_bins, orientation='horizontal')
    axHisty.grid(linestyle=':')
    axHistx.set_xlim(axScatter.get_xlim())
    axHisty.set_ylim(axScatter.get_ylim())

    plt.savefig(filename)

    plt.close('all')

    return

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


