import os
import numpy as np
from optparse import OptionParser
import skimage.color
import skimage.io
from skimage.measure import label, regionprops

#import matplotlib
#import matplotlib.pyplot as plt
#from matplotlib.colors import NoNorm
#import csv

import shutil
#from matplotlib.ticker import NullFormatter

import skimage.morphology 
from skimage.morphology import erosion, dilation, watershed
from skimage.future import graph

import pdb
import sys

import pickle 

from utils.Postprocessing import PrepareProb, HreconstructionErosion, find_maxima, ArrangeLabel
from Data.HistogramNormalization2 import normalize_multi_channel
from utils.random_utils import CheckOrCreate

from sklearn.ensemble import RandomForestClassifier

sys.path.append('/Users/twalter/pycharm/Challenge010/code')

NN_VERSIONS = ['UNetDistHistogramTW2', 'UNetHistogramTW2']

# BASE_FOLDER = '/Users/twalter/data/Challenge_010'
# BASE_POST_PROCESSING_DEBUG_FOLDER = os.path.join(BASE_FOLDER, 'post_processing_debug')

# NN_PRED_BASE_FOLDER = '/Users/twalter/data/Challenge_010/nn_out'

# NN_VERSIONS = ['UNetDistHistogramTW2', 'UNetHistogramTW2']
# PRED_VIS_FOLDER = '/Users/twalter/data/Challenge_010/prediction_vis'
# NN_ANALYSIS_PLOTS = '/Users/twalter/data/Challenge_010/nn_analysis'
# NODE_MODEL_FOLDER = '/Users/twalter/data/Challenge_010/RF_nodes'
# EDGE_MODEL_FOLDER = '/Users/twalter/data/Challenge_010/RF_edges'

#DEBUG_EDGE_CLASSIFICATION_FOLDER = '/Users/twalter/data/Challenge_010/edge_examples'

class ImageRetriever(object):

    def __init__(self, nn_folder, nn_names=None):
        self.nn_folder = nn_folder

        # the neural networks to combine
        if nn_names is None:
            self.nn_names = filter(lambda x: os.path.isdir(os.path.join(self.nn_folder, x)), os.listdir(self.nn_folder))
        else:
            self.nn_names = nn_names

    def get_training_folder(self, nn_name=None): 
        if nn_name is None:
            nn_name = self.nn_names[0] 

        # the training folder can be either Train or Validation.
        training_folder = os.path.join(self.nn_folder, nn_name, 'Train')
        if not os.path.isdir(training_folder):
            training_folder = os.path.join(self.nn_folder, nn_name, 'Validation')

        return training_folder

    def get_test_folder(self, nn_name=None):
        if nn_name is None:
            nn_name = self.nn_names[0] 
        test_folder = os.path.join(self.nn_folder, nn_name, 'Test')
        return test_folder

    def get_training_image_names(self):

        train_folder = self.get_training_folder()

        subfolders = filter(lambda x: os.path.isdir(os.path.join(train_folder, x)), os.listdir(train_folder))
        sample_folder = os.path.join(train_folder, subfolders[0])
        image_folders = filter(lambda x: os.path.isdir(os.path.join(sample_folder, x)), os.listdir(sample_folder))

        return image_folders

    def get_test_image_names(self):

        test_folder = self.get_test_folder()
        subfolders = filter(lambda x: os.path.isdir(os.path.join(test_folder, x)), os.listdir(test_folder))
        sample_folder = os.path.join(test_folder, subfolders[0])
        image_folders = filter(lambda x: os.path.isdir(os.path.join(sample_folder, x)), os.listdir(sample_folder))

        return image_folders

    def get_image_folder(self, img_name, nn_name=None, dataset='Train'):
        if nn_name is None:
            nn_name = self.nn_names[0]

        if dataset.lower() in ['train', 'validation']:
            temp_folder = self.get_training_folder(nn_name)
        else: 
            temp_folder = self.get_test_folder(nn_name)

        folders = filter(lambda x: os.path.isdir(os.path.join(temp_folder, x)), os.listdir(temp_folder))
        image_folder = os.path.join(temp_folder, folders[0], img_name)

        return image_folder

    def get_original_image(self, img_name, dataset='Train'):
        image_folder = self.get_image_folder(img_name, self.nn_names[0], dataset)
        image_name = os.path.join(image_folder, 'rgb.png')
        img = skimage.io.imread(image_name)
        if len(img.shape) > 2 and img.shape[-1] > 3:
            img = img[:,:,:3]
        return img

    def get_groundtruth_image(self, img_name, dataset='Train'):
        image_folder = self.get_image_folder(img_name, self.nn_names[0], dataset=dataset)
        image_name = os.path.join(image_folder, 'colored_bin.png')
        img = skimage.io.imread(image_name)
        if len(img.shape) > 2 and img.shape[-1] > 3:
            img = img[:,:,:3]
        return img

    def get_prediction_image(self, img_name, nn_name, dataset='Train'):
        image_folder = self.get_image_folder(img_name, nn_name, dataset=dataset)
        image_name = os.path.join(image_folder, 'colored_pred.png')
        
        img = skimage.io.imread(image_name)
        if len(img.shape) > 2 and img.shape[-1] > 3:
            img = img[:,:,:3]
        return img

    def get_prob_image(self, img_name, nn_name, dataset='Train'):

        image_folder = self.get_image_folder(img_name, nn_name, dataset=dataset)
        
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


class EdgeClassifier(object):

    def __init__(self, nn_folder, output_folder, model_folder, show_folder, nn_names=None):

        # nn_folder is the parent directory containing all nn-prediction subfolders. 
        self.nn_folder = nn_folder
        if not os.path.isdir(self.nn_folder):
            raise ValueError("No valid input path given: %s" % self.nn_folder)

        # model folder for edges
        self.output_folder = output_folder
        if not os.path.isdir(self.output_folder):
            print 'made %s' % self.output_folder
            os.makedirs(self.output_folder)

        # model folder for edges
        self.model_folder = model_folder
        if not os.path.isdir(self.model_folder):
            print 'made %s' % self.model_folder
            os.makedirs(self.model_folder)

        # show folder for edges (display of the removed edges)
        self.show_folder = show_folder
        if not os.path.isdir(self.show_folder):
            print 'made %s' % self.show_folder
            os.makedirs(self.show_folder)

        # the neural networks to combine
        if nn_names is None:
            self.nn_names = os.listdir(self.nn_folder)
        else:
            self.nn_names = nn_names

        # the classifier needs to be loaded 
        self.classifier_loaded = False

        # random forest classifier
        self.rf = RandomForestClassifier(n_estimators=500, oob_score=True, class_weight="balanced")

        # for retrieving images
        self.retriever = ImageRetriever(self.nn_folder, self.nn_names)

        # settings for post processing
        self.lamb=10
        self.p_thresh=.4


    def train(self, X=None, yvec=None, save_model=True):
        if X is None or yvec is None:
            ts = self.load_training_set()
            X = ts['X']
            yvec = ts['y']

        self.rf.fit(X, yvec)

        # report:
        print 'training succeeded. OOB accuracy : %.2f' % self.rf.oob_score_
        print 'Number of samples: %i ' % X.shape[0]
        print 'Number of features: %i' % X.shape[1]
        print 'Number of positive samples: %i (%.2f)' % (np.sum(yvec), np.float(np.sum(yvec)) / len(yvec))
        print 'Number of negative samples: %i (%.2f)' % ((len(yvec) - np.sum(yvec)), (1.0 - np.float(np.sum(yvec)) / len(yvec)))

        # save the model
        if save_model:
            fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'w')
            pickle.dump(self.rf, fp)
            fp.close()

        # the classifier is now loaded
        self.classifier_loaded = True

        return 

    # load the classifier from pickle file.
    def load_classifier(self):
        fp = open(os.path.join(self.model_folder, 'rf.pickle'), 'r')
        self.rf = pickle.load(fp)
        fp.close()
        return

    # save the training set.
    # ts is obtained by make_edge_training_set
    # ts is a dictionary with the image names as keys. 
    # In each entry, there are the edge features and the output
    def save_training_set(self, ts):

        image_names = sorted(ts.keys())

        X = np.concatenate([ts[image_name]['X'] for image_name in image_names])
        yvec = np.concatenate([ts[image_name]['y'] for image_name in image_names])
        
        training_set = {'X': X, 'y': yvec}

        # saving the training set.
        fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'w')
        pickle.dump(training_set, fp)
        fp.close()

        return

    # load the training set from pickle file.
    def load_training_set(self): 
        fp = open(os.path.join(self.model_folder, 'training_set.pickle'), 'r')
        training_set = pickle.load(fp)
        fp.close()
        return training_set

    # sigmoid distance to transform a distance image into a probability like map.
    def sigmoid_distance(self, img, a=0.73240819244540645, dist_shift=1):
        temp = img.astype(np.float)
        res = 1.0 / (1.0 + np.exp((-a) * (temp - dist_shift)))
        res[img==0] = 0
        return res

    # extracting the features from the list of features
    # the probability maps are also generated.
    def make_edge_training_set(self, image_list=None):

        if image_list is None:
            image_list = self.retriever.get_training_image_names()

        print 'collecting data for ', self.nn_names
        print 'collecting data for %i images' % len(image_list)
        res = {}
        
        for image_name in sorted(image_list):
            ground_truth_image = self.retriever.get_groundtruth_image(image_name)
            ground_truth_image = ground_truth_image[:,:,0]>0
            ground_truth_label_image = label(ground_truth_image, neighbors=4)

            #label_img = label(bin_img, neighbors=4)
            prob_map = self.make_average_probability_map(image_name, dataset='Train')

            X, yvec = self.get_edge_features_for_image(ground_truth_label_image, prob_map)
            if X is None:
                continue
            res[image_name] = {'X': X, 'y': yvec}
            print 'image %s : %i / %i' % (image_name, sum(yvec), len(yvec))
        return res

    # calculates features for the edge directly (on something similar to the watershed line)
    # the implementation is very slow, but could be enhanced easily by using cython.
    def get_wsl_features(self, ws_labels, prob_img, edges):

        res = {}
        for x,y in edges:
            res[(x,y)] = []
            res[(y,x)] = []

        for i in range(ws_labels.shape[0]):
            for j in range(ws_labels.shape[1]):

                # we are not considering 0 valued pixels (background)
                if ws_labels[i,j] == 0:
                    continue

                # the neighborhood determines the edge (if there is any)
                if (i - 1 >= 0) and (ws_labels[i-1,j] > 0) and (ws_labels[i,j] != ws_labels[i-1,j]):
                    res[(ws_labels[i,j], ws_labels[i-1,j])].append((i,j))
                if (i + 1 < ws_labels.shape[0]) and (ws_labels[i+1,j] > 0) and (ws_labels[i,j] != ws_labels[i+1,j]):
                    res[(ws_labels[i,j], ws_labels[i+1,j])].append((i,j))
                if (j - 1 >= 0) and (ws_labels[i,j-1] > 0) and (ws_labels[i,j] != ws_labels[i, j-1]):
                    res[(ws_labels[i,j], ws_labels[i, j-1])].append((i,j))
                if (j + 1 < ws_labels.shape[1]) and (ws_labels[i,j+1] > 0) and (ws_labels[i,j] != ws_labels[i, j+1]):
                    res[(ws_labels[i,j], ws_labels[i, j+1])].append((i,j))

        features = {}
        all_points = {}
        for i, (x,y) in enumerate(edges):
            # we take the set of all points (effectively removing duplicates)
            all_points[(x,y)] = set(res[(x,y)] + res[(y,x)])
            px = np.array([p[0] for p in all_points[(x,y)]])
            py = np.array([p[1] for p in all_points[(x,y)]])
            area = len(all_points[(x,y)])
            tortuosity = np.sqrt( (np.max(px) - np.min(px))**2 + (np.max(py) - np.min(py))**2 ) / np.float(area)

            features[(x,y)] = {'wsl_area': area,
                               'wsl_intensity': np.mean(prob_img[px, py]),
                               'wsl_tortuosity': tortuosity,
                              }
            if area > 1:
                features[(x,y)]['wsl_std'] = np.std(prob_img[px, py])
            else:
                features[(x,y)]['wsl_std'] = 0.0

        return features

    def apply_classifier_to_all_images(self, dataset, show_res=True):
        if dataset.lower() in ['train', 'validation']:
            all_images = self.retriever.get_training_image_names()
        else:
            all_images = self.retriever.get_test_image_names()

        out_folder = os.path.join(self.output_folder, dataset)
        if not os.path.isdir(out_folder):
            os.makedirs(out_folder)

        for image_name in all_images:
            print 'processing %s' % image_name
            ws = self.apply_classifier_to_image(image_name, dataset=dataset, show_res=True)

            image_out_folder = os.path.join(out_folder, image_name)
            if not os.path.isdir(image_out_folder):
                os.makedirs(image_out_folder)
            try:
                skimage.io.imsave(os.path.join(image_out_folder, 'ws_prediction.png'), ws)
            except:
                pdb.set_trace()
        return

    def apply_classifier_to_image(self, image_name, dataset, show_res=True):

        prob_img = self.make_average_probability_map(image_name, dataset=dataset)
        ws_labels = self.DynamicWatershedAlias2(prob_img)
        ws_out = ws_labels.copy()

        X, graph_structure = self.get_edge_features_for_image(None, prob_img)
        if X is None or graph_structure is None:
            return ws_out
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
            out_folder = os.path.join(self.show_folder, dataset)
            if not os.path.isdir(out_folder):
                os.makedirs(out_folder)
            image = self.retriever.get_original_image(image_name, dataset=dataset)
            image = image[:,:,:3]
            im_outline = dilation(ws_labels) - erosion(ws_labels)
            image[im_outline>0] = np.array([30, 120, 255])

            im_outline = dilation(ws_out) - erosion(ws_out)
            image[im_outline>0] = np.array([255, 255, 255])

            skimage.io.imsave(os.path.join(out_folder, 'edge_classifciation_%s.png' % image_name), image)

        return ws_out


    def get_edge_features_for_image(self, gt_label_img, prob_img, sim_thresh_low=.3, sim_thresh_high=.5):
        ws_labels = self.DynamicWatershedAlias2(prob_img)

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

            # calc of the joint ellipse
            coords = np.vstack([props[x-1].coords, props[y-1].coords])
            m02  = np.sum(coords[:,0]**2) - np.sum(coords[:,0])**2 / coords.shape[0]
            m20  = np.sum(coords[:,1]**2) - np.sum(coords[:,1])**2 / coords.shape[0]
            m11 = np.sum(coords[:,0] * coords[:,1]) - (np.sum(coords[:,0]) * np.sum(coords[:,1])) / coords.shape[0]
            eigenvalue_1 = .5 * (m02 + m20) + .5 * np.sqrt(4 * m11 * m11 + (m20 - m02)**2)
            eigenvalue_2 = .5 * (m02 + m20) - .5 * np.sqrt(4 * m11 * m11 + (m20 - m02)**2)
            area_ellipse = np.pi * 4.0 / coords.shape[0] * np.sqrt(eigenvalue_1 * eigenvalue_2)
            joint_feat = np.abs(coords.shape[0] - area_ellipse) / np.clip(area_ellipse, a_min=1.0, a_max=None)

            feature_dict[(x,y)]['ellipse_diff'] = joint_feat - .5 * (ellipse_feature[x] + ellipse_feature[y])

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

    def DynamicWatershedAlias2(self, p_img):
        b_img = (p_img > self.p_thresh) + 0
        Probs_inv = PrepareProb(p_img)

        Hrecons = HreconstructionErosion(Probs_inv, self.lamb)
        markers_Probs_inv = find_maxima(Hrecons, mask = b_img)
        markers_Probs_inv = label(markers_Probs_inv)
        ws_labels = watershed(Hrecons, markers_Probs_inv, mask=b_img, watershed_line=False)

        arrange_label = ArrangeLabel(ws_labels)

        return arrange_label


    def make_average_probability_map(self, img_name, dataset='Validation'):

        # read probability maps from dist, unet and un-normalized unet
        prob_maps = {}
        for nn_name in self.nn_names:
            temp = self.retriever.get_prob_image(img_name, nn_name, dataset=dataset)
            if len(temp.shape) > 2 : 
                temp = temp[:,:,0]

            # for dist functions, calculate the sigmoid
            if nn_name.rfind('Dist') > 0:
                prob_maps[nn_name] = self.sigmoid_distance(temp)
            else:
                prob_maps[nn_name] = temp / 255.0

            # output to an image folder
            temp = 255.0 * prob_maps[nn_name]
            temp = temp.astype(np.uint8)

        all_maps = np.array([prob_maps[x] for x in self.nn_names])
        avg_prob = np.mean(all_maps, axis=0)

        return avg_prob


if __name__ == '__main__':
    parser = OptionParser()

    # input folder is the parent directory of all the nn-outputs
    parser.add_option('--input', dest="input", type="str")

    # where to write the watershed output (after the processing)
    parser.add_option('--output', dest="output", type="str")

    # where to write the watershed output (after the processing)
    parser.add_option('--modelfolder', dest="modelfolder", type="str")

    # where to write the images with the removed edges
    parser.add_option('--showfolder', dest="showfolder", type="str")

    # the names of the outputs to be combined. Should be a comma separated list
    parser.add_option('--nn_names', dest="nn_names", type="str")

    # actions: feature extraction yes/no
    parser.add_option('--feature_extraction', action='store_true')

    # actions: training yes/no
    parser.add_option('--train', action='store_true')

    # actions: predict yes/no
    parser.add_option('--predict_train', action='store_true')

    # actions: predict yes/no
    parser.add_option('--predict_test', action='store_true')

    (options, args) = parser.parse_args()

    input_folder = options.input 
    if not os.path.isdir(input_folder):
        raise ValueError("folder %s does not exist" % input_folder)

    output_folder = options.output
    CheckOrCreate(output_folder)

    model_folder = options.modelfolder
    CheckOrCreate(model_folder)

    show_folder = options.showfolder
    CheckOrCreate(show_folder)

    nn_names = [x.strip() for x in options.nn_names.split(',')]

    ec = EdgeClassifier(input_folder, output_folder, model_folder, show_folder, nn_names)


    if options.feature_extraction:
        # feature extraction
        ts = ec.make_edge_training_set()
        ec.save_training_set(ts)

    if options.train:
        ec.train()

    if options.predict_train:
        ec.apply_classifier_to_all_images('Train')

    if options.predict_test:
        ec.apply_classifier_to_all_images('Test')

    print 'DONE'

