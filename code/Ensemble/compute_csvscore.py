
from utils.EvaluationFile import WriteEvaluation
from optparse import OptionParser
from glob import glob
from skimage.io import imread
from skimage.measure import label
import pdb
def CreateDic(path):
    files = glob(path + '/*/ws_prediction.png')
    res = {}
    for f in files:
        res[f.split('/')[-2]] = imread(f)
    return res


parser = OptionParser()
parser.add_option('--input', dest="input", type="str")
parser.add_option('--output', dest="output", type="str")
(options, args) = parser.parse_args()



res = CreateDic(options.input)

WriteEvaluation(options.output, res)