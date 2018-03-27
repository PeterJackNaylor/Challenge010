# -*- coding: utf-8 -*-
import time
import pdb
from optparse import OptionParser
from UNetValidation import Model_pred
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
from utils.UsefulFunctionsCreateRecord import GatherFiles, LoadRGB_GT_QUEUE
from utils.metrics import AJI_fast, DataScienceBowlMetrics
from utils.random_utils import CheckOrCreate, UNetAugment
from utils.Postprocessing import PostProcess
from skimage.io import imread, imsave
from skimage import measure, img_as_ubyte
from UNetTest import GetHP
import os

def GetLastCheckPoint(LOG):
    from glob import glob
    files = [int(el.split('.meta')[0].split('ckpt-')[-1]) for el in glob(LOG + '/*.meta')]
    return max(files)

def initialize_uninitialized_vars(sess):
    from itertools import compress
    global_vars = tf.global_variables() + tf.local_variables()
    is_not_initialized = sess.run([~(tf.is_variable_initialized(var))
                                   for var in global_vars])
    not_initialized_vars = list(compress(global_vars, is_not_initialized))

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars)) 

def remove_elmement(l1, l2):
    """
    not useful
    """
    l3 = [x for x in l1 if x not in l2]
    return l3

def h(row, rgb_g, bbg_g):
    """
    Number of groups in rgb and bbg
    """
    if row["WhiteBackGround"] == 1:
        return 0
    elif row["BlackBackGround"] == 1:
        return row["background_BBG"] + 1
    else:
        return bbg_g + 1 + row["background_RGB"]
# maybe the + 1 is not useful

def PrepDomainTrainTable(info, values, list_img_test):
    info = pd.read_csv(info, index_col=0)
    g = lambda x: x.split('.')[0]
    info.index = info.index.map(g)
    val = pd.read_csv(values)
    f = lambda x: x['path'].split('/')[-1]
    f1 = lambda x: x.split('/')[-1]
    val["path"] = val.apply(f, axis=1)
    val = val.set_index("path")
    test_ = pd.DataFrame({"path":[g(f1(el)) for el in list_img_test]})
    test_["test"] = 1
    test_ = test_.set_index("path")

    res = pd.concat([info, val, test_], axis=1)
    res["test"] = res["test"].fillna(0)
    rgb_g = res["background_RGB"].max() + 1
    bbg_g = res["background_BBG"].max() + 1
    res["Group"] = res.apply(lambda row: h(row, rgb_g, bbg_g), axis=1)

    return res

def ComputeDomainScore(tab, latest_score, to_improve):
   tmp = tab.copy()
   tmp = tmp[tmp["Group"].isin(to_improve)]
   tmp = tmp[tmp["test"] == 1]
   group = tmp.groupby("Group")
   domain_scores = group.mean()["DSB"]
   r_domain_scores = domain_scores.loc[to_improve]
   worst = r_domain_scores.argmin()
   return worst, domain_scores

def GeneratePossibleImages(tab, var_name, list_img, dic, mod):
    tab = tab[tab['train'] == 1][tab['test'] == 1]
    wgt_tab = 1 - tab.groupby('Group').mean()[var_name]
    #domain_int = int(tab.groupby('Group').mean()[var_name].argmin())
    #tab = tab[tab['Group'] == domain_int]
    possible_img = tab.index
    small_list_img = [el for el in list_img if el.split('/')[-1].split('.')[0] in possible_img]
    imgs = []
    labs = []
    weights = []
    for img in small_list_img:
        ind = os.path.basename(img).replace('.png', '')
        gr = tab.loc[ind, "Group"]
        w = float(wgt_tab.loc[gr])
        for sub_img, sub_lab in LoadRGB_GT_QUEUE(img, dic, mod.IMAGE_SIZE[0], mod.IMAGE_SIZE, True):
            imgs.append(sub_img)
            labs.append(sub_lab)
            weights.append(w)
    return imgs, labs, weights
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed
    
# @timeit
def ComputeScore(name, dic, mod, step):
    key = [k for k in dic.keys() if name in k][0]
    pred_mod = mod.pred(key) 
    lbl_path = dic[key]
    lbl = imread(lbl_path)
    S = PostProcess(pred_mod[0,:,:,1], mod.P1, mod.P2)
    G = measure.label(lbl)
#    imsave(name + "check_{}.png".format(step), img_as_ubyte(pred_mod[0,:,:,1]))
    if S.max() > G.max() * 10:
        score = 0
    else:
        score = DataScienceBowlMetrics(G, S)[0]
    return score

def ValidationDomaine(tab, var_name, list_img, dic, mod, step):
    tab = tab[tab['train'] == 1][tab['test'] == 1]
    f = lambda x: ComputeScore(x.name, dic, mod, step)
    tab[var_name] = tab['DSB'].copy()
    tab['DSB'] = tab.apply(f, axis=1)
    return tab
def GenerateFeedDic(imgs, labs, weights, bs, mod):
    n = 92
    imgs = np.array(imgs)
    labs = np.array(labs)
    n_el = len(imgs)
    rand_ord = np.arange(n_el)
    np.random.shuffle(rand_ord)
    # maybe add mean subtraction
    batch_img = np.zeros(shape=(bs, mod.IMAGE_SIZE[0] + 2 * n , mod.IMAGE_SIZE[1] + 2 * n, 3), dtype='float')
    batch_lbl = np.zeros(shape=(bs, mod.IMAGE_SIZE[0], mod.IMAGE_SIZE[1], 1), dtype='float')
    batch_wgt = np.zeros(shape=(bs, 1, 1, 1), dtype='float')
    for i in range(0, n_el, bs):
        if i + bs > n_el:
            index = rand_ord[i::]
            index_ = np.random.choice(rand_ord, bs - len(index))
            index = np.concatenate([index, index_])
        else:
            index = rand_ord[i:(i + bs)]
        for k, j in enumerate(index):
            batch_img[k] = imgs[j] - mod.MEAN_NPY
            batch_lbl[k, :, :, 0] = labs[j, n:-n, n:-n]
            batch_wgt[k, 0, 0, 0] = weights[j]
        yield {mod.input_node:batch_img, mod.train_labels_node:batch_lbl,
                                         mod.batch_sample_weight: batch_wgt}


class Model2(Model_pred):
    def init_training_graph(self):

        with tf.name_scope('Evaluation'):
            self.logits = self.conv_layer_f(self.last, self.logits_weight, strides=[1,1,1,1], scope_name="logits/")
            self.predictions = tf.argmax(self.logits, axis=3)
            init_weight = np.reshape(np.ones(self.BATCH_SIZE, dtype='float32'), [self.BATCH_SIZE,1,1,1])
            self.batch_sample_weight = tf.placeholder_with_default(init_weight,
                                                                    shape=[self.BATCH_SIZE, 1, 1, 1])

            self.weighted_logits = self.logits * self.batch_sample_weight

            with tf.name_scope('Loss'):
                self.loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.weighted_logits,
                                                                          labels=tf.squeeze(tf.cast(self.train_labels_node, tf.int32), squeeze_dims=[3]),
                                                                          name="entropy")))
                tf.summary.scalar("entropy", self.loss)

            with tf.name_scope('Accuracy'):

                LabelInt = tf.squeeze(tf.cast(self.train_labels_node, tf.int64), squeeze_dims=[3])
                CorrectPrediction = tf.equal(self.predictions, LabelInt)
                self.accuracy = tf.reduce_mean(tf.cast(CorrectPrediction, tf.float32))
                tf.summary.scalar("accuracy", self.accuracy)

            with tf.name_scope('Prediction'):

                self.TP = tf.count_nonzero(self.predictions * LabelInt)
                self.TN = tf.count_nonzero((self.predictions - 1) * (LabelInt - 1))
                self.FP = tf.count_nonzero(self.predictions * (LabelInt - 1))
                self.FN = tf.count_nonzero((self.predictions - 1) * LabelInt)

            with tf.name_scope('Precision'):

                self.precision = tf.divide(self.TP, tf.add(self.TP, self.FP))
                tf.summary.scalar('Precision', self.precision)

            with tf.name_scope('Recall'):

                self.recall = tf.divide(self.TP, tf.add(self.TP, self.FN))
                tf.summary.scalar('Recall', self.recall)

            with tf.name_scope('F1'):

                num = tf.multiply(self.precision, self.recall)
                dem = tf.add(self.precision, self.recall)
                self.F1 = tf.scalar_mul(2, tf.divide(num, dem))
                tf.summary.scalar('F1', self.F1)

            with tf.name_scope('MeanAccuracy'):
                
                Nprecision = tf.divide(self.TN, tf.add(self.TN, self.FN))
                self.MeanAcc = tf.divide(tf.add(self.precision, Nprecision) ,2)
                tf.summary.scalar('Performance', self.MeanAcc)
            #self.batch = tf.Variable(0, name = "batch_iterator")

            self.train_prediction = tf.nn.softmax(self.logits)

            self.test_prediction = tf.nn.softmax(self.logits)

        tf.global_variables_initializer().run()

        print('Computational graph initialised')

    def retrain(self, list_img, dic, summary_train, output_csv):
        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        # self.global_step = tf.Variable(GetLastCheckPoint(self.LOG), trainable=False)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)

        initialize_uninitialized_vars(self.sess)
        self.regularize_model()

        # self.Saver()
        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS
        print "self.global step", int(self.global_step.eval())
        begin = int(self.global_step.eval())
        print "begin", begin
        for step in range(begin, steps + begin):  
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            l_d = 0
            count = 0
            All_Images, All_lbl, All_wgt = GeneratePossibleImages(summary_train, "DSB", list_img, dic, self)
            minimum_number_of_updates = 50
            if len(All_Images) < minimum_number_of_updates:
                mini_epoch = minimum_number_of_updates // len(All_Images)
	    else:
                mini_epoch = 1
            for __ in range(mini_epoch):
                for f_d in GenerateFeedDic(All_Images, All_lbl, All_wgt, self.BATCH_SIZE, self):
                    _, l, lr, predictions, s = self.sess.run(
                                [self.training_op, self.loss, self.learning_rate,
                                 self.train_prediction, self.merged_summary], feed_dict=f_d)
                    l_d += l
                    count += 1
            l = l_d / count 
            i = datetime.now()
            print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
            self.summary_writer.add_summary(s, step)                
            print('  Step %d of %d' % (step, steps))
            print('  Learning rate: %.10f \n') % lr
            print('  Mini-batch loss: %.5f \n ') % l
            summary_train = ValidationDomaine(summary_train, "DSB_{}".format(step), list_img, dic, self, step)
        self.saver.save(self.sess, self.LOG + '/' + "retrain-model.ckpt", step)
        summary_train.to_csv('retraining.csv')
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
    parser.add_option('--table', dest="table", type="str")
    parser.add_option('--info', dest="info", type="str")
    parser.add_option('--hp', dest="hp", type="str")

    (options, args) = parser.parse_args()


    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
    P1, P2 = GetHP(options.hp)
    # LEARNING_RATE = options.lr
    BATCH_SIZE = options.bs
    SIZE = (options.size_train, options.size_train)
    # samples_per_epoch = len([0 for record in tf.python_io.tf_record_iterator(options.TFRecord)] )
    N_ITER_MAX = options.epoch 
    LRSTEP = "10epoch"
    N_TRAIN_SAVE = -1
    LOG = options.log
    WEIGHT_DECAY = options.wd 
    N_FEATURES = int(options.log.split('__')[-2])
    WEIGHT_DECAY = float(options.log.split('__')[-3])
    LEARNING_RATE = float(options.log.split('__')[-4]) / 100
    N_EPOCH = options.epoch
    N_THREADS = options.THREADS
    MEAN_FILE = options.mean_file 
    test_val = int(LOG.split('fold-')[-1])
    list_img, dic = GatherFiles(options.path, test_val , "test")
    table = PrepDomainTrainTable(options.info, options.table, list_img)
    # DROPOUT = options.dropout
    # drop out is unused
    model = Model2("",                 LEARNING_RATE=LEARNING_RATE,
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
                                       DROPOUT=0.5,
                                       EARLY_STOPPING=10)
    model.P1 = P1
    model.P2 = P2
    output_name = LOG + ".csv"
    model.retrain(list_img, dic, table, output_name)
    
