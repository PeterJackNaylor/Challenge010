# -*- coding: utf-8 -*-
from utils.metrics import ComputeMetrics
from glob import glob
from Nets.UNetBatchNorm import UNetBatchNorm
import tensorflow as tf
import numpy as np
import os
from os.path import abspath
from utils.random_utils import CheckOrCreate, UNetAugment, UNetAdjust
from utils.UsefulFunctionsCreateRecord import GatherFiles
from utils.TensorflowDataGen import read_and_decode
from skimage.io import imread
import math
from datetime import datetime
import pdb
from optparse import OptionParser
import pandas as pd
import os

class Model(UNetBatchNorm):
    def init_queue(self, tfrecords_filename):
        with tf.device('/cpu:0'):
            self.init_data, self.image, self.annotation = read_and_decode(tfrecords_filename,
                                                                      self.IMAGE_SIZE[0], 
                                                                      self.IMAGE_SIZE[1],
                                                                      self.BATCH_SIZE,
                                                                      self.N_THREADS,
                                                                      self.NUM_CHANNELS)

        print("Queue initialized")

    def test(self, p1, p2, steps):
        loss, roc = 0., 0.
        acc, F1, recall = 0., 0., 0.
        precision, jac, AJI = 0., 0., 0.
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.Saver()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(steps):  
            feed_dict = {self.is_training: False} 
            l,  prob, batch_labels = self.sess.run([self.loss, self.train_prediction,
                                                               self.train_labels_node], feed_dict=feed_dict)
            loss += l
            out = ComputeMetrics(prob[0,:,:,1], batch_labels[0,:,:,0], p1, p2)
            acc += out[0]
            roc += out[1]
            jac += out[2]
            recall += out[3]
            precision += out[4]
            F1 += out[5]
            AJI += out[6]
        coord.request_stop()
        coord.join(threads)
        loss, acc, F1 = np.array([loss, acc, F1]) / steps
        recall, precision, roc = np.array([recall, precision, roc]) / steps
        jac, AJI = np.array([jac, AJI]) / steps
        return loss, acc, F1, recall, precision, roc, jac, AJI

    def Validation(self, list_img, dic, step, early_stoping_max=10):
        l, acc, F1, recall, precision, meanacc = [], [], [], [], [], []
        for img_path in list_img:
            img = imread(img_path)[:,:,0:4].astype("float")
            img -= self.MEAN_NPY
            img = UNetAdjust(img)
            img = UNetAugment(img)
            label = imread(dic[img_path])
            label = UNetAdjust(label)
            Xval = img[np.newaxis, :]
            Yval = label[np.newaxis, :]
            Yval = Yval[:, :, :, np.newaxis]
            feed_dict = {self.input_node: Xval,
                         self.train_labels_node: Yval,
                         self.is_training: False}
            l_tmp, acc_tmp, F1_tmp, recall_tmp, precision_tmp, meanacc_tmp, s = self.sess.run([self.loss, 
                                                                                        self.accuracy, self.F1,
                                                                                        self.recall, self.precision,
                                                                                        self.MeanAcc, self.merged_summary], feed_dict=feed_dict)
            l.append(l_tmp)
            acc.append(acc_tmp)
            F1.append(F1_tmp)
            recall.append(recall_tmp)
            precision.append(precision_tmp)
            meanacc.append(meanacc_tmp)
        l = np.mean([el if not math.isnan(el)else 0. for el in l])
        acc = np.mean([el if not math.isnan(el)else 0. for el in acc])
        F1 = np.mean([el if not math.isnan(el) else 0. for el in F1])
        recall = np.mean([el if not math.isnan(el) else 0. for el in recall])
        precision = np.mean([el if not math.isnan(el) else 0. for el in precision])
        meanacc = np.mean([el if not math.isnan(el) else 0.5 for el in meanacc])

        summary = tf.Summary()
        summary.value.add(tag="TestMan/Accuracy", simple_value=acc)
        summary.value.add(tag="TestMan/Loss", simple_value=l)
        summary.value.add(tag="TestMan/F1", simple_value=F1)
        summary.value.add(tag="TestMan/Recall", simple_value=recall)
        summary.value.add(tag="TestMan/Precision", simple_value=precision)
        summary.value.add(tag="TestMan/Performance", simple_value=meanacc)
        self.summary_test_writer.add_summary(summary, step) 

        self.summary_test_writer.add_summary(s, step) 
        print('  Validation loss: %.1f' % l)
        print('       Accuracy: %1.f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % (acc * 100, meanacc * 100, recall * 100, precision * 100, F1 * 100))
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)
        wgt_path = self.LOG + '/' + "model.ckpt-{}".format(step)
        return l, acc, F1, recall, precision, meanacc, wgt_path

    def train(self, list_img, dic, output_csv):
        data_res = pd.DataFrame()

        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH

        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)

        trainable_var = tf.trainable_variables()
        
        self.regularize_model()
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS

        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.sess.run(self.init_data)
        early_finish = False
        for step in range(steps):      
            # print "saving images"
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])
	    #pdb.set_trace()
            # from skimage.io import imsave
            #for i in range(Xval.shape[0]):
            #    for j in range(Xval.shape[3]):
            #        img = (Xval + self.MEAN_NPY).astype('uint8')
            #        imsave('step_{}_n_{}_chan_{}_.png'.format(step, i, j), img[i,:,:,j])
            if step % self.N_PRINT == 0:
                if step != 0:
                    i = datetime.now()
                    print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                    self.summary_writer.add_summary(s, step)                
                    error, acc, acc1, recall, prec, f1 = self.error_rate(predictions, batch_labels, step)
                    print('  Step %d of %d' % (step, steps))
                    print('  Learning rate: %.5f \n') % lr
                    print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n       acc1: %.1f%% \n       recall: %1.f%% \n       prec: %1.f%% \n       f1 : %1.f%% \n' % 
                         (l, acc, acc1, recall, prec, f1))
                    l, acc, F1, recall, precision, meanacc, wgt_path = self.Validation(list_img, dic, step)
                    data_res.loc[step, "loss"] = l
                    data_res.loc[step, "acc"] = acc
                    data_res.loc[step, "F1"] = F1
                    data_res.loc[step, "recall"] = recall
                    data_res.loc[step, "precision"] = precision
                    data_res.loc[step, "meanacc"] = meanacc
                    data_res.loc[step, "wgt_path"] = abspath(wgt_path)
                    if self.early_stopping(data_res, "F1"):
                        best_wgt = np.array(data_res["wgt_path"])[-(self.early_stopping_max + 1)]
                        make_it_seem_new = self.LOG + '/' + "model.ckpt-{}".format(step+10)
                        os.symlink(best_wgt + ".data-00000-of-00001" ,make_it_seem_new + ".data-00000-of-00001")
                        os.symlink(best_wgt + ".index" ,make_it_seem_new + ".index")
                        os.symlink(best_wgt + ".meta" ,make_it_seem_new + ".meta")
                        early_finish = True
                        break
        if not early_finish:
            best_wgt = np.array(data_res["wgt_path"])[-(self.early_stopping_max + 1)]
            make_it_seem_new = self.LOG + '/' + "model.ckpt-{}".format(step+10)
            os.symlink(best_wgt + ".data-00000-of-00001" ,make_it_seem_new + ".data-00000-of-00001")
            os.symlink(best_wgt + ".index" ,make_it_seem_new + ".index")
            os.symlink(best_wgt + ".meta" ,make_it_seem_new + ".meta")
        data_res.to_csv(output_csv)
        


if __name__== "__main__":
    parser = OptionParser()
    parser.add_option('--tf_record', dest="TFRecord", type="str")
    parser.add_option('--learning_rate', dest="lr", type="float")
    parser.add_option('--batch_size', dest="bs", type="int")
    parser.add_option('--log', dest="log", type="str")
    parser.add_option('--weight_decay', dest="wd", type="float")
    parser.add_option('--n_features', dest="nfeat", type="int")

    parser.add_option('--path', dest="path", type="str")
    parser.add_option('--test', dest="test", type="int")

    parser.add_option('--size_train', dest="size_train", type="int")
    parser.add_option('--split', dest="split", type="str")
    parser.add_option('--unet', dest="unet", type="int")
    parser.add_option('--seed', dest="seed", type="int")
    parser.add_option('--epoch', dest="epoch", type="int")
    parser.add_option('--threads', dest="THREADS", default=50, type="int")
    parser.add_option('--mean_file', dest="mean_file", type="str")

    (options, args) = parser.parse_args()


    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    SIZE = (options.size_train, options.size_train)
    samples_per_epoch = len([0 for record in tf.python_io.tf_record_iterator(options.TFRecord)] )
    N_ITER_MAX = options.epoch * samples_per_epoch // BATCH_SIZE ## defined later
    LRSTEP = "10epoch"
    if options.epoch == 1:
        N_TRAIN_SAVE = samples_per_epoch // BATCH_SIZE // 5
    else:
        N_TRAIN_SAVE = samples_per_epoch // BATCH_SIZE
    LOG = options.log
    WEIGHT_DECAY = options.wd 
    N_FEATURES = options.nfeat
    N_EPOCH = options.epoch
    N_THREADS = options.THREADS
    MEAN_FILE = options.mean_file 
    # DROPOUT = options.dropout
    # drop out is unused
    model = Model(TFRecord,            LEARNING_RATE=LEARNING_RATE,
                                       BATCH_SIZE=BATCH_SIZE,
                                       IMAGE_SIZE=SIZE,
                                       NUM_LABELS=2,
                                       NUM_CHANNELS=4,
                                       STEPS=N_ITER_MAX,
                                       LRSTEP=LRSTEP,
                                       N_PRINT=N_TRAIN_SAVE,
                                       LOG=LOG,
                                       SEED=42,
                                       WEIGHT_DECAY=WEIGHT_DECAY,
                                       N_FEATURES=N_FEATURES,
                                       N_EPOCH=N_EPOCH,
                                       N_THREADS=N_THREADS,
                                       MEAN_FILE=MEAN_FILE,
                                       DROPOUT=0.5,
                                       EARLY_STOPPING=20)

    list_img, dic = GatherFiles(options.path, options.test, "test")
    output_name = LOG + ".csv"
    model.train(list_img, dic, output_name)
    


