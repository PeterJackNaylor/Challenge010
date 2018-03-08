from optparse import OptionParser
import pandas as pd
from utils.random_utils import CheckOrCreate, generate_wsl
from sklearn.model_selection import StratifiedKFold
from os.path import join, basename
from os import symlink
from skimage.io import imread, imsave
from scipy.ndimage.morphology import distance_transform_cdt

DOMAIN_VARIABLE = ["background_RGB", "background_BBG", "WhiteBackGround"]


def Create_Key_PerGroup(tab, list_var):
    
    def f(r, l):
        if r["WhiteBackGround"] == 1:
            return "0"
        elif r["RGB"] == 1:
            return str(r["background_RGB"] + 1)
        else:
            return str(r["background_BBG"] + 1 + 6)
        val = [str(int(r[el])) for el in l]
        return "_".join(val)

    tab["group"] = tab.apply(lambda row: f(row, list_var), axis=1)

def Symlink_Mask(r, dst_rgb, dst_mask):
    rgb = r['path_to_image']
    #print rgb
    mask_name = r['path_to_label']
    symlink(rgb, join(dst_rgb, basename(rgb)))
    mask = imread(mask_name)
    line = generate_wsl(mask)
    mask[mask > 0] = 1
    mask[line > 0] = 0
    mask = distance_transform_cdt(mask)
    imsave(join(dst_mask, basename(mask_name)), mask)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--input', dest="input", type="str")
    parser.add_option('--output', dest="output", type="str")
    parser.add_option('--splits', dest="splits", type="int")
    (options, args) = parser.parse_args()

    table = pd.read_csv(options.input, index_col=0)
    train = table[table["train"] == 1]
    test = table[table["train"] == 0]
    CheckOrCreate(options.output)
    Create_Key_PerGroup(train, DOMAIN_VARIABLE)

    skf = StratifiedKFold(n_splits=options.splits)
    k = 0
    Fold_k_S = join(options.output, 'Slide_{}')
    Fold_k_M = join(options.output, 'GT_{}')  
    for train_index, test_index in skf.split(train, train['group']):
        test_fold = train.ix[test_index] 

        CheckOrCreate(Fold_k_S.format(k))    
        CheckOrCreate(Fold_k_M.format(k))   
        test_fold.apply(lambda row: Symlink_Mask(row, Fold_k_S.format(k), Fold_k_M.format(k)), axis=1)
        k += 1