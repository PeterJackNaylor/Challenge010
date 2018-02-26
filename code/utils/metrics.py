from skimage.measure import label
from skimage.io import imread
import numpy as np
import pdb
import time
from progressbar import ProgressBar
from sklearn.metrics import confusion_matrix
pbar = ProgressBar()

def Intersection(A, B):
    C = A + B
    C[C != 2] = 0
    C[C == 2] = 1
    return C

def Union(A, B):
    C = A + B
    C[C > 0] = 1
    return C


def AssociatedCell(G_i, S):
    def g(indice):
        S_indice = np.zeros_like(S)
        S_indice[ S == indice ] = 1
        NUM = float(Intersection(G_i, S_indice).sum())
        DEN = float(Union(G_i, S_indice).sum())
        return NUM / DEN
    res = map(g, range(1, S.max() + 1))
    indice = np.array(res).argmax() + 1
    return indice

def AJI(G, S):
    """
    AJI as described in the paper, AJI is more abstract implementation but 100times faster
    """
    G = label(G, background=0)
    S = label(S, background=0)

    C = 0
    U = 0 
    USED = np.zeros(S.max())

    for i in pbar(range(1, G.max() + 1)):
        only_ground_truth = np.zeros_like(G)
        only_ground_truth[ G == i ] = 1
        j = AssociatedCell(only_ground_truth, S)
        only_prediction = np.zeros_like(S)
        only_prediction[ S == j ] = 1
        C += Intersection(only_prediction, only_ground_truth).sum()
        U += Union(only_prediction, only_ground_truth).sum()
        USED[j - 1] = 1

    def h(indice):
        if USED[indice - 1] == 1:
            return 0
        else:
            only_prediction = np.zeros_like(S)
            only_prediction[ S == indice ] = 1
        return only_prediction.sum()
    U_sum = map(h, range(1, S.max() + 1))
    U += np.sum(U_sum)
    return float(C) / float(U)  



pbar2 = ProgressBar()
def AJI_fast(G, S):
    G = label(G, background=0)
    S = label(S, background=0)
    if S.sum() == 0:
        return 0.
    C = 0
    U = 0 
    USED = np.zeros(S.max())

    G_flat = G.flatten()
    S_flat = S.flatten()
    G_max = np.max(G_flat)
    S_max = np.max(S_flat)
    m_labels = max(G_max, S_max) + 1
    cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
    LIGNE_J = np.zeros(S_max)
    for j in range(1, S_max + 1):
        LIGNE_J[j - 1] = cm[:, j].sum()

    for i in range(1, G_max + 1):
        LIGNE_I_sum = cm[i, :].sum()
        def h(indice):
            LIGNE_J_sum = LIGNE_J[indice - 1]
            inter = cm[i, indice]

            union = LIGNE_I_sum + LIGNE_J_sum - inter
            return inter / union
        
        JI_ligne = map(h, range(1, S_max + 1))
        best_indice = np.argmax(JI_ligne) + 1
        C += cm[i, best_indice]
        U += LIGNE_J[best_indice - 1] + LIGNE_I_sum - cm[i, best_indice]
        USED[best_indice - 1] = 1

    U_sum = ((1 - USED) * LIGNE_J).sum()
    U += U_sum
    return float(C) / float(U)  

def ComputeMetrics(**args):
    print "not implemented"

def DataScienceBowlMetrics(G, S):
    def ComputeAssociations(G, S):
        G = label(G, background=0)
        S = label(S, background=0)

        G_flat = G.flatten()
        S_flat = S.flatten()
        G_max = np.max(G_flat)
        S_max = np.max(S_flat)
        m_labels = max(G_max, S_max) + 1
        cm = confusion_matrix(G_flat, S_flat, labels=range(m_labels)).astype(np.float)
        cm_like = np.zeros_like(cm)
        for line_ind in range(G_max + 1):
            LINE_SUM = cm[line_ind,:].sum()
            for col_ind in range(S_max + 1):
                inter = cm[line_ind, col_ind]
                COLUMN_SUM = cm[:, col_ind].sum()
                union = LINE_SUM + COLUMN_SUM - inter
                cm_like[line_ind, col_ind] = float(inter) / float(union)
        return cm_like[1:(G_max + 1), 1:(S_max + 1)]
    def precision_at(IOU, threshold):
        HITS = IOU > threshold
        tp_ = np.sum(np.sum(HITS, axis=1) == 1)
        fp_ = np.sum(np.sum(HITS, axis=0) == 0)
        fn_ = np.sum(np.sum(HITS, axis=1) == 0)
        return tp_, fp_, fn_

    iou = ComputeAssociations(G, S)
    p = []
    TP = []
    FN = []
    FP = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(iou, t)
        TP.append(tp)
        FP.append(fp)
        FN.append(fn)
        score_t = float(tp) / (tp + fp + fn)
        p.append(score_t)

    return np.mean(p), p, TP, FN, FP