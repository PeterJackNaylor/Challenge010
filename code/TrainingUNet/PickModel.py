

from glob import glob
import numpy as np
import pandas as pd
from utils.random_utils import CheckOrCreate

def GetNames():
    files = np.unique([el.split('__fold')[0] for el in glob('*__fold-*.csv')])
    return {f:glob(f+"__fold*.csv") for f in files}

def ComputeScore(list_csv, var_name):
    scores = np.zeros(len(list_csv), dtype="float")
    epoch_number = np.zeros(len(list_csv), dtype="float")
    for i, csv in enumerate(list_csv):
        tables = pd.read_csv(csv, index_col=0)
        scores[i] = tables[[var_name]].max()
        epoch_number[i] = tables.shape[0] - 10
    return np.concatenate([scores, epoch_number])


if __name__ == '__main__':
    hyper_param_test = GetNames() 
    dic = {}
    for key in hyper_param_test.keys():
        dic[key] = ComputeScore(hyper_param_test[key], "F1")
    n = len(hyper_param_test[key])
    tab = pd.DataFrame.from_dict(dic, orient='index')
    mean = tab[[i for i in range(n)]].mean(axis=1)
    std = tab[[i for i in range(n)]].std(axis=1)
    tab["mean"] = mean
    tab["std"] = std
    best = mean.argmax()
    CheckOrCreate(best)
    tab.to_csv('test_tables.csv')