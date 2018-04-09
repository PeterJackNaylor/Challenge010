# -*- coding: utf-8 -*-
from UNet3Validation import Model_pred, MultiClassToBinMap
from glob import glob
import tensorflow as tf
import numpy as np
import os
from os.path import join, basename, abspath
from utils.random_utils import CheckOrCreate, UNetAugment, UNetAdjust_pixel, sliding_window, color_bin
from utils.EvaluationFile import WriteEvaluation
from Data.patch_img import Overlay_with_pred
from skimage.io import imread, imsave
from optparse import OptionParser
import pandas as pd
import os
from skimage import img_as_ubyte
from skimage.morphology import remove_small_objects
import pdb
from skimage.measure import label


def GetHP(csv_path):
    table = pd.read_csv(csv_path, index_col=0)
    ind = table.idxmax()
    ind = ind[0][1:-1].split(', ')
    ind = [float(el) for el in ind]
    return ind


if __name__== "__main__":
    parser = OptionParser()
    parser.add_option('--hp', dest="hp", type="str")
    parser.add_option('--name', dest="name", type="str")
    parser.add_option('--mean_file', dest="mean_file", type="str")
    parser.add_option('--output_sample', dest="output_sample", type="str")
    parser.add_option('--output_csv', dest="output_csv", type="str")

    (options, args) = parser.parse_args()

    N_FEATURES = int(options.name.split('__')[-1])
    P1, P2 = GetHP(options.hp)
    MEAN_FILE = options.mean_file 
    outcsv = options.output_csv
    FILES = glob('*.png')
    MODELS = glob(options.name + "__fold*")
    dic = {__:[] for __ in FILES}

    for LOG in MODELS:
        model = Model_pred("", BATCH_SIZE=1,
                               IMAGE_SIZE=(212, 212),
                               NUM_LABELS=2,
                               NUM_CHANNELS=4,
                               LOG=LOG,
                               N_FEATURES=N_FEATURES,
                               N_THREADS=50,
                               MEAN_FILE=MEAN_FILE)
        for __ in FILES:
            prediction = model.pred(__)
            dic[__].append(prediction)
        tf.reset_default_graph() # so that it can restore properly the next model

    dic_final_pred = {}
    dic_prob = {}
    CheckOrCreate(options.output_sample)
    for key in dic.keys():
        OUT_ID = join(options.output_sample, basename(key).replace('.png', ''))
        CheckOrCreate(OUT_ID)
        dic_prob[key] = np.mean(np.concatenate(dic[key]), axis=0)[:,:,1] 
        dic_final_pred[key] = MultiClassToBinMap(dic_prob[key], P2)
        # dic_final_pred[key] = (dic_prob[key] > P2).astype('uint8')
        dic_final_pred[key] = label(dic_final_pred[key])
        dic_final_pred[key] = remove_small_objects(dic_final_pred[key], 32)
        img_mean = np.mean(imread(key)[:,:,0:3])
        if img_mean < 125:
            color_cont = False
        else:
            color_cont = True
        #put rgb image
        os.symlink(abspath(key), join(OUT_ID, "rgb.png"))            
        imsave(join(OUT_ID, "colored_pred.png"), color_bin(dic_final_pred[key])) 
        imsave(join(OUT_ID, "output_DNN_mean.png"), img_as_ubyte(dic_prob[key]))
        for k, el in enumerate(dic[key]):
            imsave(join(OUT_ID, "output_DNN_{}.png").format(k), img_as_ubyte(el[0,:,:,1])) 
        imsave(join(OUT_ID, "contours_pred.png"), Overlay_with_pred(key, dic_final_pred[key], color_cont).astype('uint8'))




    WriteEvaluation(outcsv, dic_final_pred)
