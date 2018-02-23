# -*- coding: utf-8 -*-
from glob import glob
from UNet import Model
import tensorflow as tf
import numpy as np
import os
from os.path import abspath, join, basename
from utils.random_utils import CheckOrCreate, UNetAugment, UNetAdjust_pixel, sliding_window, color_bin
from Data.patch_img import Overlay, Overlay_with_pred
from utils.Postprocessing import PostProcess
from utils.metrics import AJI_fast
from skimage.io import imread, imsave
from optparse import OptionParser
import pandas as pd
from utils.UsefulFunctionsCreateRecord import GatherFiles
from skimage import img_as_ubyte
from skimage.measure import label
import pdb
from sklearn.metrics import f1_score
from progressbar import ProgressBar

P1_List = range(6, 14)
P2_list = [0.5]

class Model_pred(Model):
    def pred(self, img_path): 
        img = imread(img_path)[:,:,0:3].astype("float")
        x, y, z = img.shape
        img -= self.MEAN_NPY
        stepSize = UNetAdjust_pixel(img)
        windowSize = (184 + stepSize[0], 184 + stepSize[1])
        img = UNetAugment(img)
        result = np.zeros(shape=(1, x, y, 2), dtype='float')
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

def ComputeScores(list_rgb, dic_gt, dic_prob, 
                  p1, p2, keep_memory=False,
                  path_save='./tmp'):
    res_AJI = []
    res_F1 = []
    for path in list_rgb:
        GT = imread(dic_gt[path])
        S = PostProcess(dic_prob[path][:,:,1], p1, p2)
        res_AJI.append(AJI_fast(GT, S))
        res_F1.append(ComputeF1(GT, S))
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
            imsave(join(OUT, "colored_pred.png"), color_bin(S)) 
            imsave(join(OUT, "output_DNN.png"), img_as_ubyte(dic_prob[path][:,:,1]))
            imsave(join(OUT, "contours_gt.png"), Overlay(path, dic_gt[path], color_cont).astype('uint8')) 
            imsave(join(OUT, "contours_pred.png"), Overlay_with_pred(path, S, color_cont).astype('uint8'))
#            pdb.set_trace()
    if keep_memory:
        return res_AJI, res_F1
    else:
        return np.mean(res_AJI), np.mean(res_F1)


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
                               NUM_LABELS=2,
                               NUM_CHANNELS=3,
                               N_FEATURES=N_FEATURES,
                               MEAN_FILE=MEAN_FILE)
        number_test = int(mod.split('-')[-1])
        test_images, dic_test_gt = GatherFiles(options.path, number_test, split="test")
        test_images = test_images
        test_img_all += test_images
        dic_test_gt_all = dict(dic_test_gt_all, **dic_test_gt)

        pbar = ProgressBar()
        for img_test_path in pbar(test_images):
            dic_pred[img_test_path] = model.pred(img_test_path)[0]
        tf.reset_default_graph() # so that it can restore properly the next model

    HP_dic = {}
    for p1 in P1_List:
        for p2 in P2_list:
            HP_dic[(p1, p2)] = ComputeScores(test_img_all, dic_test_gt_all, dic_pred, p1, p2)[0]
    
    tab = pd.DataFrame.from_dict(HP_dic, orient='index')
    tab.to_csv('Hyper_parameter_selection.csv')
    P1, P2 = tab[0].idxmax()
    CheckOrCreate(options.output)
    aji__, f1__ = ComputeScores(test_img_all, dic_test_gt_all, dic_pred, p1, p2, True, options.output)
    pathsss = [join(options.output, basename(path).replace('.png', '')) for path in test_img_all]
    tab_values = pd.DataFrame.from_dict({'path':pathsss, 'F1':f1__, 'AJI':aji__})
    tab_values.to_csv(join(options.output, '__summary_per_image.csv'), index=False)
