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

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten()==1)[0] # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b+1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def WriteEvaluation(name, dic):
    f = open(name, 'w')
    f.write("ImageId,EncodedPixels\n")
    for key in dic.keys():
        img_lab = label(dic[key].T) .flatten()
        for i in range(1, img_lab.max() + 1):
            # one_nuclei = img_lab.copy()
            # one_nuclei[one_nuclei != i] = 0
            # one_nuclei[one_nuclei == i] = 1
            # encoding_pixel = summarise_line(one_nuclei)
            encoding_pixel = rle_encoding(img_lab==i)
            encoding_pixel_string = [str(el) for el in encoding_pixel]
            f.write('{},{}\n'.format(key.replace('.png', ''), " ".join(encoding_pixel_string)))
    f.close()