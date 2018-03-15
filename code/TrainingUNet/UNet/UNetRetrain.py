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
    domain_int = int(tab.groupby('Group').mean()[var_name].argmin())
    tab = tab[tab['Group'] == domain_int]
    possible_img = tab.index
    small_list_img = [el for el in list_img if el.split('/')[-1].split('.')[0] in possible_img]
    imgs = []
    labs = []
    for img in small_list_img:
        for sub_img, sub_lab in LoadRGB_GT_QUEUE(img, dic, mod.IMAGE_SIZE[0], mod.IMAGE_SIZE, True):
            imgs.append(sub_img)
            labs.append(sub_lab)
    return imgs, labs
def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print '%r %2.2f sec' % \
              (method.__name__, te-ts)
        return result

    return timed
    
@timeit
def ComputeScore(name, dic, mod):
    key = [k for k in dic.keys() if name in k][0]
    pred_mod = mod.pred(key) 
    lbl_path = dic[key]
    lbl = imread(lbl_path)
    S = PostProcess(pred_mod[0,:,:,1], mod.P1, mod.P2)
    G = measure.label(lbl)
    imsave(name + "check.png", img_as_ubyte(pred_mod[0,:,:,1]))
    return DataScienceBowlMetrics(G, S)[0]

def ValidationDomaine(tab, var_name, list_img, dic, mod):
    tab = tab[tab['train'] == 1][tab['test'] == 1]
    f = lambda x: ComputeScore(x.name, dic, mod)
    tab[var_name] = tab['DSB'].copy()
    tab['DSB'] = tab.apply(f, axis=1)
    return tab
def GenerateFeedDic(imgs, labs, bs, mod):
    n = 92
    imgs = np.array(imgs)
    labs = np.array(labs)
    n_el = len(imgs)
    rand_ord = np.arange(n_el)
    np.random.shuffle(rand_ord)
    # maybe add mean subtraction
    batch_img = np.zeros(shape=(bs, mod.IMAGE_SIZE[0] + 2 * n , mod.IMAGE_SIZE[1] + 2 * n, 3), dtype='float')
    batch_lbl = np.zeros(shape=(bs, mod.IMAGE_SIZE[0], mod.IMAGE_SIZE[1], 1), dtype='float')
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
        yield {mod.input_node:batch_img, mod.train_labels_node:batch_lbl}


class Model2(Model_pred):
    def retrain(self, list_img, dic, summary_train, output_csv):
        epoch = self.STEPS * self.BATCH_SIZE // self.N_EPOCH
        self.Saver()
        trainable_var = tf.trainable_variables()
        self.LearningRateSchedule(self.LEARNING_RATE, self.K, epoch)
        self.optimization(trainable_var)
        self.ExponentialMovingAverage(trainable_var, self.DECAY_EMA)
        uninitialized_vars = []
	for var in tf.global_variables():
    	    try:
                self.sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.initialize_variables(uninitialized_vars)
        
        # init_op = tf.group(tf.global_variables_initializer(),
        #            tf.local_variables_initializer())
        self.sess.run(init_new_vars_op)
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
            All_Images, All_lbl = GeneratePossibleImages(summary_train, "DSB", list_img, dic, self)
            minimum_number_of_updates = 50
            if len(All_Images) < minimum_number_of_updates:
                mini_epoch = minimum_number_of_updates // len(All_Images)
	    else:
                mini_epoch = 1
            for __ in range(mini_epoch):
                for f_d in GenerateFeedDic(All_Images, All_lbl, self.BATCH_SIZE, self):
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
            print('  Learning rate: %.5f \n') % lr
            print('  Mini-batch loss: %.5f \n ') % l
            print('  Max value: %.5f \n ') % np.max(predictions)
            summary_train = ValidationDomaine(summary_train, "DSB_{}".format(step), list_img, dic, self)
            pdb.set_trace()
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
    model.P1 = P1
    model.P2 = P2
    output_name = LOG + ".csv"
    model.retrain(list_img, dic, table, output_name)
    
