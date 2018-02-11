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
from skimage.io import imread
import math
from datetime import datetime
import pdb
from optparse import OptionParser
import pandas as pd


class Model(UNetBatchNorm):
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

    def Validation(self, list_img, dic, step):
        l, acc, F1, recall, precision, meanacc = 0., 0., 0., 0., 0., 0.
        for img_path in list_img:
            img = imread(img_path)[:,:,0:3].astype("float")
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
                                                                                        self.MeanAcc,
                                                                                        self.merged_summary], feed_dict=feed_dict)
            l += l_tmp
            acc += acc_tmp
            F1 += F1_tmp
            recall += recall_tmp
            precision += precision_tmp
            meanacc += meanacc_tmp

        l, acc, F1, recall, precision, meanacc = np.array([l, acc, F1, recall, precision, meanacc]) / len(list_img)

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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for step in range(steps):      
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            print "train", step
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])

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

        coord.request_stop()
        coord.join(threads)
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
    parser.add_option('--threads', dest="THREADS", default=100, type="int")
    parser.add_option('--mean_file', dest="mean_file", type="str")

    (options, args) = parser.parse_args()


    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
    LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    SIZE = (options.size_train, options.size_train)
    N_ITER_MAX = 10 ## defined later
    LRSTEP = "10epoch"
    N_TRAIN_SAVE = 2
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
                                       NUM_CHANNELS=3,
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
                                       DROPOUT=0.5)
    if SPLIT == "train":
        list_img, dic = GatherFiles(options.path, options.test, "test")
        output_name = LOG + ".csv"
        model.train(list_img, dic, output_name)
    elif SPLIT == "test":
        p1 = options.p1
        file_name = options.output
        f = open(file_name, 'w')
        outs = model.test(options.p1, 0.5, N_ITER_MAX)
        outs = [LOG] + list(outs) + [p1, 0.5]
        NAMES = ["ID", "Loss", "Acc", "F1", "Recall", "Precision", "ROC", "Jaccard", "AJI", "p1", "p2"]
        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*NAMES))

        f.write('{},{},{},{},{},{},{},{},{},{},{}\n'.format(*outs))


