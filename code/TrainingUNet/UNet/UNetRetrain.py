# -*- coding: utf-8 -*-
import pdb
from optparse import OptionParser
from UNet import Model
import tensorflow as tf
from datetime import datetime
import numpy as np
import pandas as pd
from utils.UsefulFunctionsCreateRecord import GatherFiles, LoadRGB_GT_QUEUE

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

def GeneratePossibleImages(domain_int, tab, list_img, dic, mod):
    tab = tab[tab['test'] == 0]
    tab = tab[tab['Group'] == domain_int]
    possible_img = tab.index
    small_list_img = [el for el in list_img if el in possible_img]
    imgs = []
    labs = []
    for img in small_list_img:
        for sub_img, sub_lab in LoadRGB_GT_QUEUE(img, dic, mod.SIZE[0], mod.SIZE, True):
            imgs.append(sub_img)
            labs.append(sub_lab)
    return imgs, labs


def GenerateFeedDic(imgs, labs, bs, mod):
    imgs = np.array(imgs)
    labs = np.array(labs)
    n_el = len(imgs)
    rand_ord = np.arange(n_el)
    np.random.shuffle(rand_ord)
    batch_img = np.array(shape=(bs, mod.SIZE[0], mod.SIZE[1], 3))
    batch_lbl = np.array(shape=(bs, mod.SIZE[0], mod.SIZE[1], 1))
    for i in range(0, n_el, bs):
        if i + bs > n_el:
            index = rand_ord[i::]
            index_ = np.random.choice(rand_ord, bs - len(index))
            index = index + index_
        else:
            index = rand_ord[i:(i + bs)]
        for k, j in enumerate(index):
            batch_img[k] = imgs[j]
            batch_lbl[k] = labs[j]
        yield {mod.input_node_f:batch_img, mod.input_lbl_f:batch_lbl}


class Model2(Model):
    def retrain(self, list_img, dic, summary_train, output_csv):
        tab = ElaborateTrainStatistics(summary_train)
        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
        self.sess.run(init_op)
        self.regularize_model()

        # self.Saver()
        self.summary_test_writer = tf.summary.FileWriter(self.LOG + '/test',
                                            graph=self.sess.graph)

        self.summary_writer = tf.summary.FileWriter(self.LOG + '/train', graph=self.sess.graph)
        self.merged_summary = tf.summary.merge_all()
        steps = self.STEPS
        pdb.set_trace()
        print "self.global step", int(self.global_step.eval())
        begin = int(self.global_step.eval())
        print "begin", begin
        for step in range(begin, steps + begin):  
            # self.optimizer is replaced by self.training_op for the exponential moving decay
            l_d = 0
            count = 0
            All_Images, All_lbl = GeneratePossibleImages(domain_int, tab, list_img, dic, self)
            for f_d in GenerateFeedDic(All_Images, All_lbl, self.BATCH_SIZE, self):
                _, l, lr, predictions, s = self.sess.run(
                            [self.training_op, self.loss, self.learning_rate,
                             self.train_prediction, self.merged_summary], feed_dict=f_d)
                l_d += l
                count += 1
            l = l_d / count 
            if step % self.N_PRINT == 0:
                i = datetime.now()
                print i.strftime('%Y/%m/%d %H:%M:%S: \n ')
                self.summary_writer.add_summary(s, step)                
                print('  Step %d of %d' % (step, steps))
                print('  Learning rate: %.5f \n') % lr
                print('  Mini-batch loss: %.5f \n ') % l
                print('  Max value: %.5f \n ') % np.max(predictions)
                self.ValidationDomaine(tab)


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

    (options, args) = parser.parse_args()


    SPLIT = options.split

    ## Model parameters
    TFRecord = options.TFRecord
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
    LEARNING_RATE = float(options.log.split('__')[-4])
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

    list_img, dic = GatherFiles(options.path, options.test, "test")
    output_name = LOG + ".csv"
    model.retrain(list_img, dic, output_name)
    
