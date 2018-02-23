import numpy as np
from skimage.measure import regionprops, label
import pdb

def summarise_line(array):
    string_result = ""
    n = array.shape[0]
    test = np.zeros(shape=(2, n), dtype='int')
    for i in range(2):
        test[i] = array
    test = label(test, background=0)
    reg = regionprops(test)
    for obj in reg:
        coords = obj.coords
        min_coords = min(coords, key = lambda t:[1])[1]
        size = obj.area / 2
        string_result += str(min_coords) + " " + str(size) + " "
    return string_result

def WriteEvaluation(name, dic):
    f = open(name, 'w')
    f.write("ImageId,EncodedPixels\n")
    for key in dic.keys():
        img_lab = label(dic[key]) .flatten()
        for i in range(1, img_lab.max() + 1):
            one_nuclei = img_lab.copy()
            one_nuclei[one_nuclei != i] = 0
            one_nuclei[one_nuclei == i] = 1
            encoding_pixel = summarise_line(one_nuclei)
            f.write('{},{}\n'.format(key.replace('.png', ''), encoding_pixel))
    f.close()