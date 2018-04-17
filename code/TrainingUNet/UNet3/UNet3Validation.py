# -*- coding: utf-8 -*-
from glob import glob
from UNet3 import Model
import tensorflow as tf
import numpy as np
import os
from os.path import abspath, join, basename
from utils.random_utils import CheckOrCreate, UNetAugment, UNetAdjust_pixel, sliding_window, color_bin
from Data.patch_img import Overlay, Overlay_with_pred
from utils.metrics import AJI_fast, DataScienceBowlMetrics
from skimage.io import imread, imsave
from optparse import OptionParser
import pandas as pd
from utils.UsefulFunctionsCreateRecord import GatherFiles
from skimage import img_as_ubyte
from skimage.measure import label
from skimage.morphology import dilation
import pdb
from sklearn.metrics import f1_score
from progressbar import ProgressBar

P2_list = [0.4, 0.5, 0.6]


def MultiClassToBinMap(prob_array, p2 = 0.4):
    """
    Based on Neeraj's matlab code for post-processing
    """
    x, y, z = prob_array.shape
    boundary   = prob_array[:,:, 2]
    nucleus    = prob_array[:,:, 1] > p2
    background = prob_array[:,:, 0]
    image_bin = np.zeros(shape=(x, y))
    conComp = label(nucleus)
    current = np.zeros(conComp.max())
    bound_mean = np.mean(boundary[boundary != 0])
    selem = np.array([[0,1,0],[1,1,1],[0,1,0]])

    for i in range(1, conComp.max()+1):
        temp_bw = np.zeros(shape=(x, y))
        temp_bw[conComp == i] = 1
        prev = 0
        count = 0
        while current[i-1] < bound_mean and count < 2:
            prev = current[i-1]
            temp1 = dilation(temp_bw, selem=selem)
            count += 1
            temp_diff = (temp1 - temp_bw).astype('uint8')
            temp_mult = temp_diff * boundary
            temp_bw = temp1
        image_bin[temp_bw > 0] = i


    return image_bin

class Model_pred(Model):
    def pred(self, img_path): 
        img = imread(img_path)[:,:,0:3].astype("float")
        x, y, z = img.shape
        img -= self.MEAN_NPY
        stepSize = UNetAdjust_pixel(img)
        windowSize = (184 + stepSize[0], 184 + stepSize[1])
        img = UNetAugment(img)
        result = np.zeros(shape=(1, x, y, self.NUM_LABELS), dtype='float')
        for xb, yb, xe, ye, sub_img in sliding_window(img, stepSize, windowSize):
            Xval = sub_img[np.newaxis, :]
            feed_dict = {self.input_node: Xval,
                         self.is_training: False}
            pred = self.sess.run(self.test_prediction, feed_dict=feed_dict)
            result[0, xb:(xb + stepSize[0]), yb:(yb + stepSize[1])] = pred
        return result


def ComputeF1(G, S):
    Gc = G.copy().flatten()
    Sc = S.copy().flatten()
    Gc[Gc > 0] = 1
    Sc[Sc > 0] = 1
    return f1_score(Gc, Sc)

from skimage.morphology import dilation

def Label3(GT):
    inside = label(GT == 1)
    inside = dilation(inside)
    return inside

def ComputeScores(list_rgb, dic_gt, dic_prob, 
                  p2, keep_memory=False,
                  path_save='./tmp'):
    res_AJI = []
    res_F1 = []
    res_DSB = []
    res_ps = []
    res_TP = []
    res_FN = []
    res_FP = []
    for path in list_rgb:
        GT = imread(dic_gt[path])
        GT = Label3(GT)
        S = MultiClassToBinMap(dic_prob[path], p2)
        res_AJI.append(AJI_fast(GT, S))
        res_F1.append(ComputeF1(GT, S))
        scores, p_s, TP, FN, FP = DataScienceBowlMetrics(GT, S)
        res_DSB.append(scores)
        res_ps.append(p_s)
        res_TP.append(TP)
        res_FN.append(FN)
        res_FP.append(FP)
        if keep_memory:
            img_mean = np.mean(imread(path)[:,:,0:3])
            if img_mean < 125:
                color_cont = False
            else:
                color_cont = True
            OUT = join(path_save, basename(path).replace('.png', ''))
            CheckOrCreate(OUT)
            os.symlink(abspath(path), join(OUT, "rgb.png"))
            os.symlink(abspath(dic_gt[path]), join(OUT, "bin.png"))
            imsave(join(OUT, "colored_bin.png"), color_bin(label(GT)))
            imsave(join(OUT, "colored_pred.png"), color_bin(S.astype(int))) 
            imsave(join(OUT, "contours_gt.png"), Overlay(path, dic_gt[path], color_cont).astype('uint8')) 
            imsave(join(OUT, "output_class0.png"), img_as_ubyte(dic_prob[path][:,:,0]))
            imsave(join(OUT, "output_class1.png"), img_as_ubyte(dic_prob[path][:,:,1]))
            imsave(join(OUT, "output_class2.png"), img_as_ubyte(dic_prob[path][:,:,2]))
            # imsave(join(OUT, "contours_pred.png"), Overlay_with_pred(path, S, color_cont).astype('uint8'))
#            pdb.set_trace()
    if keep_memory:
        return res_AJI, res_F1, res_DSB, res_ps, res_TP, res_FN, res_FP
    else:
        return np.mean(res_AJI), np.mean(res_F1), np.mean(res_DSB)


if __name__== "__main__":
    parser = OptionParser()
    parser.add_option('--name', dest="name", type="str")
    parser.add_option('--mean_file', dest="mean_file", type="str")
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--path', dest="path", type="str")

    (options, args) = parser.parse_args()

    NAME = options.name
    N_FEATURES = int(NAME.split('__')[-1])
    MEAN_FILE = options.mean_file 
    outputname = options.output
    POSSIBLE_MODELS = [el.replace('.csv', '') for el in glob(NAME + "*.csv")]
    test_img_all = []
    dic_test_gt_all = {}
    dic_pred = {}

    for mod in POSSIBLE_MODELS:
        model = Model_pred("", IMAGE_SIZE=(212, 212),
                               LOG=mod,
                               NUM_LABELS=3,
                               NUM_CHANNELS=3,
                               N_FEATURES=N_FEATURES,
                               MEAN_FILE=MEAN_FILE)
        number_test = int(mod.split('-')[-1])
        test_images, dic_test_gt = GatherFiles(options.path, number_test, split="test")
        test_img_all += test_images
        dic_test_gt_all = dict(dic_test_gt_all, **dic_test_gt)

        pbar = ProgressBar()
        for img_test_path in pbar(test_images):
            dic_pred[img_test_path] = model.pred(img_test_path)[0]
        tf.reset_default_graph() # so that it can restore properly the next model

    HP_dic = {}
    p1 = 0
    for p2 in P2_list:    
        HP_dic[(p1, p2)] = ComputeScores(test_img_all, dic_test_gt_all, dic_pred, p2)
    tab = pd.DataFrame.from_dict(HP_dic, orient='index')
    tab.columns = ['AJI', 'F1', 'DSB']
    tab.to_csv('Hyper_parameter_selection.csv')
    P1, P2 = tab["DSB"].idxmax()
    CheckOrCreate(options.output)
    aji__, f1__, DSB__, ps__, tp__, fn__, fp__ = ComputeScores(test_img_all, dic_test_gt_all, dic_pred, p2, True, options.output)
    ps__, tp__, fn__, fp__ = [np.array(el) for el in [ps__, tp__, fn__, fp__]]
    pathsss = [join(options.output, basename(path).replace('.png', '')) for path in test_img_all]
    df_dic = {'path':pathsss, 'F1':f1__, 'AJI':aji__, 'DSB':DSB__}
    for k, t in enumerate(np.arange(0.5, 1, 0.05)):
        df_dic['precision_t_{}'.format(t)] = ps__[:, k]
        df_dic['tp_t_{}'.format(t)] = tp__[:, k]
        df_dic['fn_t_{}'.format(t)] = fn__[:, k]
        df_dic['fp_t_{}'.format(t)] = fp__[:, k]
    tab_values = pd.DataFrame.from_dict(df_dic)

    tab_values.to_csv('__summary_per_image.csv', index=False)
