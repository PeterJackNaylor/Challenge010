import os
import numpy as np
import fnmatch
import skimage.color
import skimage.io
from skimage.measure import label, regionprops
from skimage.future import graph
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm
import csv

import shutil
from matplotlib.ticker import NullFormatter

import skimage.morphology 
from skimage.morphology import erosion, dilation
import pdb
import sys

import pickle 

from utils.Postprocessing import PrepareProb, HreconstructionErosion, find_maxima, ArrangeLabel
from skimage.morphology import watershed

sys.path.append('/Users/twalter/pycharm/Challenge010/code')

from Data.HistogramNormalization2 import normalize_multi_channel
from sklearn.ensemble import RandomForestClassifier

BASE_FOLDER = '/Users/twalter/data/Challenge_010'
BASE_POST_PROCESSING_DEBUG_FOLDER = os.path.join(BASE_FOLDER, 'post_processing_debug')

NN_PRED_BASE_FOLDER = '/Users/twalter/data/Challenge_010/nn_out'

NN_VERSIONS = ['UNetDistHistogramTW2', 'UNetHistogramTW2']
PRED_VIS_FOLDER = '/Users/twalter/data/Challenge_010/prediction_vis'
NN_ANALYSIS_PLOTS = '/Users/twalter/data/Challenge_010/nn_analysis'
NODE_MODEL_FOLDER = '/Users/twalter/data/Challenge_010/RF_nodes'
EDGE_MODEL_FOLDER = '/Users/twalter/data/Challenge_010/RF_edges'

DEBUG_EDGE_CLASSIFICATION_FOLDER = '/Users/twalter/data/Challenge_010/edge_examples'

def get_all_images(dataset='Validation'):
    nn_version='Dist'
    temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, dataset)
    folders = filter(lambda x: x[0] != '.' and os.path.isdir(os.path.join(temp_folder, x)), os.listdir(temp_folder))
    sample_folder = os.path.join(temp_folder, folders[0])
    image_folders = filter(lambda x: x[0] != '.' and os.path.isdir(os.path.join(sample_folder, x)), os.listdir(sample_folder))
    return image_folders

def get_nn_pred_folder(nn_version='ContrastUNet', dataset='Validation'):
    temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, dataset)
    if not os.path.isdir(temp_folder) and dataset == 'Validation':
        temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, 'Train')

    folders = filter(lambda x: os.path.isdir(os.path.join(temp_folder, x)), os.listdir(temp_folder))
    res_folder = os.path.join(temp_folder, folders[0])
    return res_folder

def get_nn_test_folder(nn_version='ContrastUNet'):
    temp_folder = os.path.join(NN_PRED_BASE_FOLDER, nn_version, 'Test')
    folders = filter(lambda x: x[0] != '.' and os.path.isdir(os.path.join(temp_folder, x)), os.listdir(temp_folder))
    res_folder = os.path.join(temp_folder, folders[0])
    return res_folder

def get_image_folder(img_name, nn_version='ContrastUNet', dataset='Validation'): 
    res_folder = os.path.join(get_nn_pred_folder(nn_version, dataset=dataset), img_name)
    return res_folder

def get_original_image(img_name, nn_version='ContrastUNet', dataset='Validation'):
    image_folder = get_image_folder(img_name, nn_version, dataset=dataset)
    image_name = os.path.join(image_folder, 'rgb.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_groundtruth_image(img_name, nn_version='ContrastUNet', dataset='Validation'):
    image_folder = get_image_folder(img_name, nn_version, dataset=dataset)
    image_name = os.path.join(image_folder, 'colored_bin.png')
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_prediction_image(img_name, nn_version='ContrastUNet', dataset='Validation'):
    image_folder = get_image_folder(img_name, nn_version, dataset=dataset)
    image_name = os.path.join(image_folder, 'colored_pred.png')
    
    img = skimage.io.imread(image_name)
    if len(img.shape) > 2 and img.shape[-1] > 3:
        img = img[:,:,:3]
    return img

def get_prob_image(img_name, nn_version='ContrastUNet', dataset='Validation'):

    image_folder = get_image_folder(img_name, nn_version, dataset=dataset)
    filenames = filter(lambda x: x.rfind('output_DNN') >= 0 , os.listdir(image_folder))
    if len(filenames) == 0:
        raise ValueError("no output_DNN files found in %s" % image_folder)
    elif len(filenames) ==1 : 
        filename = filenames[0]
    else:
        filename = 'output_DNN_mean.png'

    image_name = os.path.join(image_folder, filename)
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

def get_image_features(nn_label_img, nn_probmap):
    
    res = {}
    for nn_lab_name in nn_label_img.keys():
        label_img = nn_label_img[nn_lab_name]
        
        for nn_name in nn_probmap:
            im_prob = nn_probmap[nn_name]
            props = skimage.measure.regionprops(label_img, im_prob)
            
            intensity = np.array([obj.mean_intensity for obj in props])
            moment20 = np.array([obj.weighted_moments_normalized[2,0] for obj in props])
            moment02 = np.array([obj.weighted_moments_normalized[0,2] for obj in props])
            moment11 = np.array([obj.weighted_moments_normalized[1,1] for obj in props])

            for feature_name, vec in zip(['intensity_%s', 'intensity_moment20_%s', 'intensity_moment02_%s', 'intensity_moment11_%s'],
                                         [intensity, moment20, moment02, moment11]):
                feature_name_loaded = feature_name % nn_name
                if not feature_name_loaded in res:
                    res[feature_name_loaded] = vec
                else:
                    res[feature_name_loaded] = np.concatenate([res[feature_name_loaded], vec])

        solidity = np.array([obj.solidity for obj in props])    
        area = np.array([obj.area for obj in props])
    
    
        eigenvalue_1 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) + 
                                 .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                 for obj in props])
        eigenvalue_2 = np.array([.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) - 
                                 .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                 for obj in props])
        ellipse_area = np.pi * 4.0 * np.sqrt(eigenvalue_1 * eigenvalue_2) / area.astype(np.float)

        ellipse_feature = np.abs(area.astype(np.float) - ellipse_area) / np.clip(ellipse_area, a_min=1.0, a_max=None)
        
        for feature_name, vec in zip(['solidity', 'area', 'ellipse_feature'],
                                     [solidity, area, ellipse_feature]):
            if not feature_name in res:
                res[feature_name] = vec
            else:
                res[feature_name] = np.concatenate([res[feature_name], vec])

    return res


def get_tp_and_fp(img_ground_truth, img_prediction_label, coverage = 0.5):
    img_ground_truth_bin = img_ground_truth > 0
    props = skimage.measure.regionprops(img_prediction_label, img_ground_truth_bin)            
    intensity = np.array([obj.mean_intensity for obj in props])
    return (intensity > coverage)

def get_training_set(image_list=None, nn_names=None):
    if nn_names is None:
        nn_names = NN_VERSIONS[:2]
    if image_list is None:
        image_list = get_all_images()
    print 'collecting data for ', nn_names
    res = {}
    
    for image_name in sorted(image_list)[:3]:
        ground_truth_image = get_groundtruth_image(image_name)
        ground_truth_image = ground_truth_image[:,:,0]>0
        #label_img = label(bin_img, neighbors=4)

        nn_probmap = {}
        nn_label_img = {}
        yvec = np.array([], dtype=np.uint8)
        for nn_name in nn_names:
            temp = get_prediction_image(image_name, nn_name)
            bin_img = temp[:,:,0]>0
            nn_label_img[nn_name] = label(bin_img, neighbors=4)
            nn_probmap[nn_name] = get_prob_image(image_name, nn_name)
            #pdb.set_trace()
            yvec = np.concatenate([yvec, get_tp_and_fp(ground_truth_image, nn_label_img[nn_name], coverage=0.5).astype(np.uint8)])

        res[image_name] = get_image_features(nn_label_img, nn_probmap)
        res[image_name]['y'] = yvec
        print 'image %s : %i / %i' % (image_name, sum(yvec), len(yvec))
    return res
 
def make_design_matrix(training_dict, image_list=None):
    if image_list is None:
        image_list = sorted(training_dict.keys())
    features = [
                'area',
                'solidity',
                'ellipse_feature',
                'intensity_ContrastUNet',
                'intensity_Dist',
                'intensity_moment02_ContrastUNet', 
                'intensity_moment20_ContrastUNet',
                'intensity_moment11_ContrastUNet',
                'intensity_moment02_Dist',
                'intensity_moment20_Dist',
                'intensity_moment11_Dist'
                ]
    X = np.vstack([np.hstack([training_dict[image][feature] for image in image_list]) for feature in features]).T
    yvec = np.hstack([training_dict[image]['y'] for image in image_list])
    
    return

class RF_nodes(object):

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight="balanced")
        self.features = [
                         'area',
                         'solidity',
                         'ellipse_feature',
                         'intensity_ContrastUNet',
                         'intensity_Dist',
#                         'intensity_moment02_ContrastUNet', 
#                         'intensity_moment20_ContrastUNet',
#                         'intensity_moment11_ContrastUNet',
#                         'intensity_moment02_Dist',
#                         'intensity_moment20_Dist',
#                         'intensity_moment11_Dist'
                       ]
        self.nn_names = ['ContrastUNet', 'Dist']
        self.model_folder = NODE_MODEL_FOLDER
        if not os.path.isdir(self.model_folder):
            os.makedirs(self.model_folder)
        self.coverage = 0.5

    def get_tp_and_fp(self, img_ground_truth, img_prediction_label, coverage = 0.5):
        img_ground_truth_bin = img_ground_truth > 0
        props = skimage.measure.regionprops(img_prediction_label, img_ground_truth_bin)            
        intensity = np.array([obj.mean_intensity for obj in props])
        return (intensity > coverage)

    def get_training_dict(self, image_list=None):

        if image_list is None:
            image_list = get_all_images()

        nn_names = self.nn_names
        print 'collecting data for ', nn_names

        res = {}        
        for image_name in sorted(image_list):
            ground_truth_image = get_groundtruth_image(image_name)
            ground_truth_image = ground_truth_image[:,:,0]>0

            nn_probmap = {}
            nn_label_img = {}
            yvec = np.array([], dtype=np.uint8)
            for nn_name in nn_names:
                temp = get_prediction_image(image_name, nn_name)
                bin_img = temp[:,:,0]>0
                nn_label_img[nn_name] = label(bin_img, neighbors=4)
                nn_probmap[nn_name] = get_prob_image(image_name, nn_name)
                yvec = np.concatenate([yvec, self.get_tp_and_fp(ground_truth_image, nn_label_img[nn_name], coverage=self.coverage).astype(np.uint8)])

            res[image_name] = get_image_features(nn_label_img, nn_probmap)
            res[image_name]['y'] = yvec
            print 'image %s : %i / %i' % (image_name, sum(yvec), len(yvec))
        return res
    
    def make_design_matrix(self, training_dict, image_list=None):
        if image_list is None:
            image_list = sorted(training_dict.keys())
        X = np.vstack([np.hstack([training_dict[image][feature] for image in image_list]) for feature in self.features]).T
        yvec = np.hstack([training_dict[image]['y'] for image in image_list])
        return X, yvec

    def get_training_set(self):
        training_dict = self.get_training_dict()
        X, yvec = self.make_design_matrix(training_dict)

        return X, yvec

    def train(self, X, yvec, save_model=True):
        self.rf.fit(X, yvec)

        # report:
        print 'training succeeded. OOB accuracy : %.2f' % self.rf.oob_score_
        print 'Number of samples: %i ' % X.shape[0]
        print 'Number of features: %i' % X.shape[1]
        print 'Number of positive samples: %i (%.2f)' % (np.sum(yvec), np.float(np.sum(yvec)) / len(yvec))
        print 'Number of negative samples: %i (%.2f)' % ((len(yvec) - np.sum(yvec)), (1.0 - np.float(np.sum(yvec)) / len(yvec)))

        if save_model:
            training_set = {'X': X, 'y': yvec}
            fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'w')
            pickle.dump(training_set, fp)
            fp.close()

            fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'w')
            pickle.dump(self.rf, fp)
            fp.close()
        return 

    def __call__(self):
        X, yvec = self.get_training_set()
        self.train(X, yvec, True)


def get_ground_truth_features(image_list=None):
    temp_folder = os.path.join(NN_ANALYSIS_PLOTS, 'bad_ex')
    if not os.path.isdir(temp_folder):
        os.makedirs(temp_folder)

    if image_list is None:
        image_list = get_all_images()
    res = {}
    for image_name in sorted(image_list):
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

def find_bad_performer(info=None):
    if info is None:
        info = read_nn_performance()
    images = sorted(info.keys())
    f1 = np.array([[np.float(info[x][nn]['F1']) for nn in NN_VERSIONS] for x in images]).T
    aji = np.array([[np.float(info[x][nn]['AJI']) for nn in NN_VERSIONS] for x in images]).T
    indices = np.where((aji[0] < 0.6)*(f1[0] > 0.8))
    bad_performers = np.array(images)[indices[0]]
    for i, bad_performer in enumerate(bad_performers):
        print '%s:\tAJI=%.3f, F1=%.3f' % (bad_performer, aji[0][indices[0][i]], f1[0][indices[0][i]])
        display_prediction(bad_performer, aji[:,indices[0][i]], f1[:,indices[0][i]])
    return

def display_prediction(img_name, aji_vec=None, f1_vec=None, out_folder=None):
    if out_folder is None:
        out_folder = PRED_VIS_FOLDER
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    im_orig = get_original_image(img_name)
    im_gt = get_groundtruth_image(img_name)

    fig, axs = plt.subplots(3, 3, tight_layout=True,
                            figsize=(15, 15))

    # original images (color)
    axs[0, 0].imshow(im_orig[:,:,0], cmap="gray")
    axs[0,0].axis('off')
    axs[0,0].set_title('Original', fontsize=26)
    axs[1, 0].imshow(im_gt, norm=NoNorm())
    axs[1,0].axis('off')
    axs[1,0].set_title('Ground Truth', fontsize=26)

    axs[2, 0].axis('off')
    #axs[0, 0].get_xaxis().set_visible(False)
    #axs[0, 0].get_yaxis().set_visible(False)

    for i in range(3):
        nn_version = NN_VERSIONS[i]
        prediction = get_prediction_image(img_name, nn_version)
        prob = get_prob_image(img_name, nn_version)
        axs[i, 1].imshow(prediction, cmap="gray", norm=NoNorm())
        axs[i, 1].axis('off')
        axs[i, 1].set_title('F1=%.2f AJI=%.2f' % (f1_vec[i], aji_vec[i]), fontsize=26)
        #axs[i, 1].set_ylabel(nn_version, size='large')
        
        axs[i, 2].imshow(prob, cmap="gray") #, norm=NoNorm())
        axs[i, 2].axis('off')
        axs[i, 2].set_title('%s: post-prob' % NN_VERSIONS[i], fontsize=26)

        #axs[i,2].annotate(nn_version, xy=(1, 0.5), xytext=(-axs[i,2].yaxis.labelpad - pad, 0),
        #                  xycoords=axs[i,2].yaxis.label, textcoords='offset points',
        #                  size='large', ha='right', va='center', rotation=90)
            
            #ax.set_ylabel(nn_version, size='large')

    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    plt.savefig(os.path.join(out_folder, '%s_comparison.png' % img_name))
    plt.close('all')

    return 

from Data.patch_img import Overlay, Overlay_with_pred

class EdgeClassifier(object):

    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight="balanced")
        self.nn_names = NN_VERSIONS
        self.model_folder = EDGE_MODEL_FOLDER
        if not os.path.isdir(self.model_folder):
            print 'made %s' % self.model_folder
            os.makedirs(self.model_folder)
        self.classifier_loaded = False
        #self.coverage = 0.5

    def train(self, X=None, yvec=None, save_model=True):
        self.rf.fit(X, yvec)

        # report:
        print 'training succeeded. OOB accuracy : %.2f' % self.rf.oob_score_
        print 'Number of samples: %i ' % X.shape[0]
        print 'Number of features: %i' % X.shape[1]
        print 'Number of positive samples: %i (%.2f)' % (np.sum(yvec), np.float(np.sum(yvec)) / len(yvec))
        print 'Number of negative samples: %i (%.2f)' % ((len(yvec) - np.sum(yvec)), (1.0 - np.float(np.sum(yvec)) / len(yvec)))

        if save_model:
            fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'w')
            pickle.dump(self.rf, fp)
            fp.close()
        self.classifier_loaded = True
        return 

    def load_classifier(self):
        fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'r')
        self.rf = pickle.load(fp)
        fp.close()
        return

    def save_training_set(self, ts):
        image_names = sorted(ts.keys())
        X = np.concatenate([ts[image_name]['X'] for image_name in image_names])
        yvec = np.concatenate([ts[image_name]['y'] for image_name in image_names])
        
        training_set = {'X': X, 'y': yvec}
        fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'w')
        pickle.dump(training_set, fp)
        fp.close()
        return

    def load_training_set(self): 
        fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'r')
        training_set = pickle.load(fp)
        fp.close()
        return training_set

    def sigmoid_distance(self, img, a=0.73240819244540645, dist_shift=1):
        temp = img.astype(np.float)
        #temp = temp / np.max(temp)
        res = 1.0 / (1.0 + np.exp((-a) * (temp - dist_shift)))
        res[img==0] = 0
        return res


    def make_edge_training_set(self, image_list=None, nn_names=None):
        if nn_names is None:
            nn_names = NN_VERSIONS
        if image_list is None:
            image_list = get_all_images()
        print 'collecting data for ', nn_names
        print 'collecting data for %i images' % len(image_list)
        res = {}
        
        for image_name in sorted(image_list):
            ground_truth_image = get_groundtruth_image(image_name)
            ground_truth_image = ground_truth_image[:,:,0]>0
            ground_truth_label_image = label(ground_truth_image, neighbors=4)

            #label_img = label(bin_img, neighbors=4)
            prob_map = self.make_average_probability_map(image_name, dataset='Validation', nn_versions=nn_names)

            X, yvec = self.get_edge_features_for_image(ground_truth_label_image, prob_map)
            if X is None:
                continue
            res[image_name] = {'X': X, 'y': yvec}
            print 'image %s : %i / %i' % (image_name, sum(yvec), len(yvec))
        return res

    def get_wsl_features(self, ws_labels, prob_img, edges):

        res = {}
        for x,y in edges:
            res[(x,y)] = []
            res[(y,x)] = []

        for i in range(ws_labels.shape[0]):
            for j in range(ws_labels.shape[1]):
                try:
                    if ws_labels[i,j] == 0:
                        continue
                    if (i - 1 >= 0) and (ws_labels[i-1,j] > 0) and (ws_labels[i,j] != ws_labels[i-1,j]):
                        res[(ws_labels[i,j], ws_labels[i-1,j])].append((i,j))
                    if (i + 1 < ws_labels.shape[0]) and (ws_labels[i+1,j] > 0) and (ws_labels[i,j] != ws_labels[i+1,j]):
                        res[(ws_labels[i,j], ws_labels[i+1,j])].append((i,j))
                    if (j - 1 >= 0) and (ws_labels[i,j-1] > 0) and (ws_labels[i,j] != ws_labels[i, j-1]):
                        res[(ws_labels[i,j], ws_labels[i, j-1])].append((i,j))
                    if (j + 1 < ws_labels.shape[1]) and (ws_labels[i,j+1] > 0) and (ws_labels[i,j] != ws_labels[i, j+1]):
                        res[(ws_labels[i,j], ws_labels[i, j+1])].append((i,j))
                except:
                    pdb.set_trace()

        #edge_labels = np.zeros(ws_labels.shape, dtype=np.int32)
        features = {}
        all_points = {}
        for i, (x,y) in enumerate(edges):
            all_points[(x,y)] = set(res[(x,y)] + res[(y,x)])
            px = np.array([p[0] for p in all_points[(x,y)]])
            py = np.array([p[1] for p in all_points[(x,y)]])
            area = len(all_points[(x,y)])
            tortuosity = np.sqrt( (np.max(px) - np.min(px))**2 + (np.max(py) - np.min(py))**2 ) / np.float(area)

            features[(x,y)] = {'wsl_area': area,
                               'wsl_intensity': np.mean(prob_img[px, py]),
                               'wsl_tortuosity': tortuosity,
                              }

        return features

    def apply_classifier_to_all_images(self):
        all_images = get_all_images()
        for image_name in all_images:
            print 'processing %s' % image_name
            self.apply_classifier_to_image(image_name)
        return

    def apply_classifier_to_image(self, image_name, nn_names=None, show_res=True):
        lamb=10
        p_thresh=.4
        if nn_names is None:
            nn_names = NN_VERSIONS
        prob_img = self.make_average_probability_map(image_name, dataset='Validation', nn_versions=nn_names)
        ws_labels = self.DynamicWatershedAlias2(prob_img, lamb, p_thresh)
        ws_out = ws_labels.copy()

        X, graph_structure = self.get_edge_features_for_image(None, prob_img)
        if X is None or graph_structure is None:
            return None
        g = graph_structure['graph']
        graph_edges = graph_structure['edges']

        if not self.classifier_loaded: 
            self.load_classifier()
        yvec = self.rf.predict(X)

        equivalence = {}
        for edge in graph_edges:
            if edge[0] not in equivalence:
                equivalence[edge[0]] = edge[0]
            if edge[1] not in equivalence:
                equivalence[edge[1]] = edge[1]

        replace_colors = dict(zip())
        for i, (x,y) in enumerate(graph_edges):
            if yvec[i] == 0.0:
                e1 = min(x,y)
                e2 = max(x,y)
                while(equivalence[e1]!=e1):
                    e1 = equivalence[e1]
                equivalence[e2] = e1
        for e2 in sorted(equivalence.keys()):
            if e2==equivalence[e2]:
                continue
            e1 = e2
            while (e1!=equivalence[e1]):
                e1 = equivalence[e1]
            ws_out[ws_out==e1] = e2

        if show_res:
            out_folder = DEBUG_EDGE_CLASSIFICATION_FOLDER
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            image = get_original_image(image_name, dataset='Validation')
            image = image[:,:,:3]            
            im_outline = dilation(ws_labels) - erosion(ws_labels)
            image[im_outline>0] = np.array([30, 120, 255])

            im_outline = dilation(ws_out) - erosion(ws_out)
            image[im_outline>0] = np.array([255, 255, 255])

            skimage.io.imsave(os.path.join(out_folder, 'edge_classifciation_%s.png' % image_name), image)

        return ws_out


    def get_edge_features_for_image(self, gt_label_img, prob_img, lamb=10, p_thresh=.4, sim_thresh_low=.3, sim_thresh_high=.5):
        ws_labels = self.DynamicWatershedAlias2(prob_img, lamb, p_thresh)

        is_training = not(gt_label_img is None)

        if is_training:
            # this is for training
            g = graph.rag_mean_color(gt_label_img, ws_labels, mode='distance', connectivity=1)
            fp_edges = filter(lambda e: (e[0] != 0) and (e[1] != 0) and (g.get_edge_data(e[0], e[1])['weight'] < sim_thresh_low), g.edges)
            tp_edges = filter(lambda e: (e[0] != 0) and (e[1] != 0) and (g.get_edge_data(e[0], e[1])['weight'] > sim_thresh_high), g.edges)
            all_edges = tp_edges + fp_edges
            graph_edges = filter(lambda x: (x[0] != 0) and (x[1] != 0), g.edges)
            print '\tkept %i out of %i edges in the training set' % (len(all_edges), len(graph_edges))
        else:
            # in this case we have no ground truth (for prediction)
            g = graph.rag_mean_color(prob_img, ws_labels, mode='distance', connectivity=1)
            graph_edges = filter(lambda x: (x[0] != 0) and (x[1] != 0), g.edges)
            all_edges = graph_edges

        if len(all_edges) == 0:
            return None, None

        # get the features of the nodes (traditional regionprops)
        props = skimage.measure.regionprops(ws_labels, prob_img)
        area = np.array([1.0] + [obj.area for obj in props])
        intensity = np.array([1.0] + [obj.mean_intensity for obj in props])
        maxval = np.array([1.0] + [obj.max_intensity for obj in props])
        eigenvalue_1 = np.array([1.0] + [.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) + 
                                 .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                 for obj in props])
        eigenvalue_2 = np.array([1.0] + [.5 * (obj.moments_central[0,2] + obj.moments_central[2,0]) - 
                                 .5 * np.sqrt(4 * obj.moments_central[1,1]**2 + (obj.moments_central[0,2] - obj.moments_central[2,0])**2)
                                 for obj in props])
        ellipse_area = np.pi * 4.0 * np.sqrt(eigenvalue_1 * eigenvalue_2) / area.astype(np.float)
        ellipse_feature = np.abs(area.astype(np.float) - ellipse_area) / np.clip(ellipse_area, a_min=1.0, a_max=None)
        solidity = np.array([1.0] + [obj.solidity for obj in props]) 
        eccentricity = np.array([1.0] + [obj.eccentricity for obj in props]) 

        # edge features (features calculated directly on the edges)
        edge_features = self.get_wsl_features(ws_labels, prob_img, graph_edges)

        # construction of the feature set
        feature_dict = {}
        for i, (x,y) in enumerate(all_edges):
            feature_dict[(x,y)] = {}
            feature_dict[(x,y)]['area_diff'] = np.abs(area[x] - area[y])
            feature_dict[(x,y)]['area_max'] = max(area[x], area[y])
            feature_dict[(x,y)]['area_min'] = min(area[x], area[y])

            feature_dict[(x,y)]['solidity_max'] = max(solidity[x], solidity[y])
            feature_dict[(x,y)]['solidity_min'] = min(solidity[x], solidity[y])

            feature_dict[(x,y)]['eccentricity_diff'] = np.abs(eccentricity[x] - eccentricity[y])
            feature_dict[(x,y)]['eccentricity_max'] = max(eccentricity[x], eccentricity[y])
            feature_dict[(x,y)]['eccentricity_min'] = min(eccentricity[x], eccentricity[y])

            feature_dict[(x,y)]['ellipse_feature_max'] = max(ellipse_feature[x], ellipse_feature[y])
            feature_dict[(x,y)]['ellipse_feature_min'] = min(ellipse_feature[x], ellipse_feature[y])

            feature_dict[(x,y)]['intensity_diff'] = np.abs(intensity[x] - intensity[y])
            feature_dict[(x,y)]['intensity_max'] = max(intensity[x], intensity[y])
            feature_dict[(x,y)]['intensity_min'] = min(intensity[x], intensity[y])

            feature_dict[(x,y)]['maxval_diff'] = np.abs(maxval[x] - maxval[y])
            feature_dict[(x,y)]['maxval_max'] = max(maxval[x], maxval[y])
            feature_dict[(x,y)]['maxval_min'] = min(maxval[x], maxval[y])

            feature_dict[(x,y)]['wsl_intensity'] = edge_features[(x,y)]['wsl_intensity']
            feature_dict[(x,y)]['wsl_tortuosity'] = edge_features[(x,y)]['wsl_tortuosity']
            feature_dict[(x,y)]['wsl_area'] = edge_features[(x,y)]['wsl_area']

        if is_training:
            yvec = np.array([1.0 for x in tp_edges] + [0.0 for x in fp_edges])
        else:
            yvec = {'graph': g, 'edges': all_edges}

        feature_names = sorted(feature_dict[all_edges[0]].keys())
        X = np.array([[feature_dict[x][feature_name] for feature_name in feature_names]
                       for x in all_edges])
        
        return X, yvec

    def DynamicWatershedAlias2(self, p_img, lamb, p_thresh = .5, debug_folder=None):
        #pdb.set_trace()
        b_img = (p_img > p_thresh) + 0
        Probs_inv = PrepareProb(p_img)

        Hrecons = HreconstructionErosion(Probs_inv, lamb)
        markers_Probs_inv = find_maxima(Hrecons, mask = b_img)
        markers_Probs_inv = label(markers_Probs_inv)
        ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img, watershed_line=False)

        arrange_label = ArrangeLabel(ws_labels)
        if not debug_folder is None:
            skimage.io.imsave(os.path.join(debug_folder, 'labels_before_processing.png'), arrange_label)

        return arrange_label


    def make_average_probability_map(self, img_name, dataset='Validation', nn_versions=None):
        if nn_versions is None:
            nn_versions = NN_VERSIONS

        # read probability maps from dist, unet and un-normalized unet
        prob_maps = {}
        for nn_version in nn_versions:
            temp = get_prob_image(img_name, nn_version, dataset=dataset)
            if len(temp.shape) > 2 : 
                temp = temp[:,:,0]

            # for dist functions, calculate the average
            if nn_version.rfind('Dist') > 0:
                prob_maps[nn_version] = self.sigmoid_distance(temp)
            else:
                prob_maps[nn_version] = temp / 255.0

            # output to an image folder
            temp = 255.0 * prob_maps[nn_version]
            temp = temp.astype(np.uint8)

        all_maps = np.array([prob_maps[x] for x in nn_versions])
        avg_prob = np.mean(all_maps, axis=0)

        return avg_prob









# TEST: 
# 9f17aea854db13015d19b34cb2022cfdeda44133323fcd6bb3545f7b9404d8ab
# 3c4c675825f7509877bc10497f498c9a2e3433bf922bd870914a2eb21a54fd26
# 0f1f896d9ae5a04752d3239c690402c022db4d72c0d2c087d73380896f72c466
def make_post_processing(img_name, dataset='Validation', nn_versions=None):
    if nn_versions is None:
        nn_versions = NN_VERSIONS

    out_folder = os.path.join(BASE_POST_PROCESSING_DEBUG_FOLDER, img_name)
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)

    # read probability maps from dist, unet and un-normalized unet
    prob_maps = {}
    for nn_version in nn_versions:
        temp = get_prob_image(img_name, nn_version, dataset=dataset)
        if len(temp.shape) > 2 : 
            temp = temp[:,:,0]

        # for dist functions, calculate the average
        if nn_version.rfind('Dist') > 0:
            prob_maps[nn_version] = sigmoid_distance(temp)
        else:
            prob_maps[nn_version] = temp / 255.0

        # output to an image folder
        temp = 255.0 * prob_maps[nn_version]
        temp = temp.astype(np.uint8)
        skimage.io.imsave(os.path.join(out_folder, 'original_prob_map_nn_%s.png' % nn_version),
                          temp)

    # for testing
    #weights = np.array([1.0, 1.0]).reshape((2, 1, 1))
    #ttt = all_maps * np.array([3.0, 2.0]).reshape((2, 1, 1))
    all_maps = np.array([prob_maps[x] for x in nn_versions])
    avg_prob = np.mean(all_maps, axis=0)
    skimage.io.imsave(os.path.join(out_folder, 'avg_prob_map.png'),
                      avg_prob)

    im_post = DynamicWatershedAlias2(avg_prob, 8, .3, out_folder)
    im_outline = (im_post > 0) + 0
    im_outline = im_outline - erosion(im_outline)
    orig_image = get_original_image(img_name, nn_version, dataset=dataset)[:,:,:3]
    orig_image[im_outline>0] = np.array([200, 100, 50])
    skimage.io.imsave(os.path.join(out_folder, 'prediction_res.png'), orig_image)
    return

def vis_helper(nn_version):
    #folder = '/Users/twalter/data/Challenge_010/nn_out/UNetHistogramTW2/Test/UNetHistogramTW1_sampleTest'
    folder = get_nn_test_folder(nn_version)

    image_folders = filter(lambda x: x[0]!='.', os.listdir(folder))
    out_folder = '/Users/twalter/data/Challenge_010/nn_test_vis'
    if not os.path.isdir(out_folder):
        os.makedirs(out_folder)
    print out_folder
    print image_folders

    for image_folder in image_folders:
        #try:
        filename = os.path.join(folder, image_folder, 'contours_pred.png')
        shutil.copyfile(filename, os.path.join(out_folder, '%s_%s.png' % (image_folder, nn_version)))
        filename = os.path.join(folder, image_folder, 'rgb.png')
        shutil.copyfile(filename, os.path.join(out_folder, '%s_rgb.png' % image_folder))
        filename = os.path.join(folder, image_folder, 'rgb.png')
        shutil.copyfile(filename, os.path.join(out_folder, '%s_rgb.png' % image_folder))
        

        #except:
        #    continue
    return



def analyse_nn_performance(out_folder=None):
    if out_folder is None:
        out_folder = NN_ANALYSIS_PLOTS
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

    info = read_nn_performance()

    # plot F1 agains DSB
    # data preparation
    images = sorted(info.keys())
    f1 = np.array([[np.float(info[x][nn]['F1']) for nn in NN_VERSIONS] for x in images]).T
    aji = np.array([[np.float(info[x][nn]['AJI']) for nn in NN_VERSIONS] for x in images]).T
    colors = ['red', 'green', 'blue']

    for i, nn in enumerate(NN_VERSIONS):
        print '%s: \tf1=%.2f\taji=%.2f' % (nn, np.mean(f1[i]), np.mean(aji[i]))

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
    plt.xlabel('AJI score of ContrastNet (unnormalized)')
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
    plt.xlabel('F1 score of ContrastNet (unnormalized)')
    plt.ylabel('F1 score of DistNet')
    plt.title('Comaprison of dist net and UNet (F1)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'Contrast_vs_Dist_F1.png'))
    plt.close('all')

    fig = plt.figure(figsize=(6,6))    
    plt.scatter(aji[2], aji[1], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('AJI score of norm-UNet (normalized)')
    plt.ylabel('AJI score of DistNet')
    plt.title('Comaprison of dist net and normalized UNet (AJI)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'NormU_vs_Dist_AJI.png'))
    plt.close('all')

    fig = plt.figure(figsize=(6,6))    
    plt.scatter(f1[2], f1[1], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('F1 score of norm-UNet (normalized)')
    plt.ylabel('F1 score of DistNet')
    plt.title('Comaprison of dist net and normalized UNet (F1)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'NormU_vs_Dist_F1.png'))
    plt.close('all')


    # plot dist against standard
    fig = plt.figure(figsize=(6,6))    
    plt.scatter(aji[0], aji[2], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('AJI score of ContrastNet (unnormalized)')
    plt.ylabel('AJI score of norm-UNet (normalized)')
    plt.title('Comaprison of ContrastNet and norm-UNet (AJI)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'Contrast_vs_normU_AJI.png'))
    plt.close('all')

    fig = plt.figure(figsize=(6,6))    
    plt.scatter(f1[0], f1[2], c='red', marker='.', s=15, edgecolor='')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel('F1 score of ContrastNet (unnormalized)')
    plt.ylabel('F1 score of norm-UNet (normalized)')
    plt.title('Comaprison of ContrastNet and norm-UNet (F1)')
    plt.plot(np.array([0, 1]), np.array([0, 1]), c='orange')
    plt.grid()
    plt.savefig(os.path.join(out_folder, 'Contrast_vs_normU_F1.png'))
    plt.close('all')

    return










def calc_joint_ellipse_feature(props, edge):
    x,y = edge
    labels = np.array([obj.label for obj in props])
    ix = labels.index(x)
    iy = labels.index(y)

    area_joint = props[ix].area + props[iy].area

    # new center:
    new_centroid = np.float(props[ix].area) / area_joint * props[ix].local_centroid + np.float(props[iy].area) / area_joint * props[iy].local_centroid
    props[ix].moments + props[iy].moments
    return



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


