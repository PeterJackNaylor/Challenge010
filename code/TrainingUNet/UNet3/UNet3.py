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
from sklearn.metrics import confusion_matrix

class Model(UNetBatchNorm):
    def init_training_graph(self):
        with tf.name_scope('Evaluation'):
            self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(self.logits, axis=3)
            
            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                          labels=tf.squeeze(tf.cast(self.train_labels_node, tf.int32), squeeze_dims=[3]),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.train_labels_node, tf.int64), squeeze_dims=[3])
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('ClassPrediction'):
                flat_LabelInt = tf.reshape(LabelInt, [-1])
                flat_predictions = tf.reshape(self.predictions, [-1])
                self.cm = tf.confusion_matrix(flat_LabelInt, flat_predictions, self.NUM_LABELS)
                flatten_confusion_matrix = tf.reshape(self.cm, [-1])
                total = tf.reduce_sum(self.cm)
                confusion_image = tf.reshape( tf.cast( self.cm, tf.float32),
                                            [1, self.NUM_LABELS, self.NUM_LABELS, 1])
                tf.summary.image('confusion', confusion_image)

            self.train_prediction = tf.nn.softmax(self.logits)

            self.test_prediction = self.train_prediction

        tf.global_variables_initializer().run()

        print('Computational graph initialised')

    def Validation(self, list_img, dic, step, early_stoping_max=10):
        l, acc = [], []
        cm = np.zeros((self.NUM_LABELS, self.NUM_LABELS))

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
            l_tmp, acc_tmp, cm_tmp, s = self.sess.run([self.loss, 
                                                        self.accuracy, self.cm,
                                                        self.merged_summary], feed_dict=feed_dict)
            l.append(l_tmp)
            acc.append(acc_tmp)
            cm += cm_tmp
        if len(list_img) != 0:
            l = np.mean([el if not math.isnan(el)else 0. for el in l])
            acc = np.mean([el if not math.isnan(el)else 0. for el in acc])


            summary = tf.Summary()
            summary.value.add(tag="TestMan/Accuracy", simple_value=acc)
            summary.value.add(tag="TestMan/Loss", simple_value=l)
            confusion = tf.Variable(cm, name='confusion' )
            confusion_image = tf.reshape( tf.cast( confusion, tf.float32),
                                          [1, self.NUM_LABELS, self.NUM_LABELS, 1])
            tf.summary.image('confusion', confusion_image)
            self.summary_test_writer.add_summary(summary, step) 

            self.summary_test_writer.add_summary(s, step) 
            print('  Validation loss: %.1f' % l)
            print('       Accuracy: %1.f%% \n \n' % (acc * 100))
        else:
            l, acc = 0, 0
            cm = np.zeros(shape=(self.NUM_LABELS, self.NUM_LABELS))
        self.saver.save(self.sess, self.LOG + '/' + "model.ckpt", step)
        wgt_path = self.LOG + '/' + "model.ckpt-{}".format(step)
        return l, acc, cm, wgt_path

    def error_rate(self, predictions, labels, iter):
        predictions = np.argmax(predictions, 3)
        labels = labels[:,:,:,0]

        cm = confusion_matrix(labels.flatten(), predictions.flatten(), labels=range(self.NUM_LABELS)).astype(np.float)
        b, x, y = predictions.shape
        total = b * x * y
        acc = cm.diagonal().sum() / total
        error = 100 - acc

        return error, acc * 100, cm


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
        #self.sess.run(self.init_data)
        early_finish = False
        CheckOrCreate("./confusion_matrix_train")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

  
        for step in range(steps):      
            # print "saving images"
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            _, l, lr, predictions, batch_labels, s = self.sess.run(
                        [self.training_op, self.loss, self.learning_rate,
                         self.train_prediction, self.train_labels_node,
                         self.merged_summary])
            if step % self.N_PRINT == 0:
                if step != 0:
                    i = datetime.now()
                    print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                    self.summary_writer.add_summary(s, step)                
                    error, acc, cm_train = self.error_rate(predictions, batch_labels, step)
                    print('  Step %d of %d' % (step, steps))
                    print('  Learning rate: %.5f \n') % lr
                    print('  Mini-batch loss: %.5f \n       Accuracy: %.1f%% \n' % 
                         (l, acc))
                    l, acc, cm, wgt_path = self.Validation(list_img, dic, step)
                    data_res.loc[step, "loss"] = l
                    data_res.loc[step, "acc"] = acc
                    data_res.loc[step, "wgt_path"] = abspath(wgt_path)
                    np.save("./confusion_matrix_train" + "/cm_{}.npy".format(step), cm_train)
                    np.save("./confusion_matrix_test" + "/cm_{}.npy".format(step), cm)
                    if self.early_stopping(data_res, "acc"):
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
                                       NUM_LABELS=3,
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
                                       DROPOUT=0.5,
                                       EARLY_STOPPING=20)

    list_img, dic = GatherFiles(options.path, options.test, "test")
    output_name = LOG + ".csv"
    model.train(list_img, dic, output_name)
    


