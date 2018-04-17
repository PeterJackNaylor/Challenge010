
from os.path import join
import os
from skimage.io import imread, imsave
import numpy as np
from utils.Postprocessing import PostProcess
from utils.random_utils import CheckOrCreate
from skimage.measure import label
import pdb
PATH = "/Users/naylorpeter/Desktop/NucleiKaggle/results/"

def CheckIfPng(l):
    found = False
    for el in l:
        if ".png" in el:
            found = True
            break
    return found

def GatherFiles(fold, phase="Train"):
    assert(phase == "Train" or phase == "Test")
    fold = join(fold, phase)
    list_ = [el for el in os.walk(fold)]
    dic = {}
    for el in list_:
        if CheckIfPng(el[2]):
            dic[ el[0].split('/')[-1] ] = {item.split('.')[0]: join(el[0], item) for item in el[2]}
    return dic

def GatherMultipleModels(list_names, phase="Train", path=PATH):
    dic = {el: GatherFiles(join(PATH, el), phase) for el in list_names}
    return dic

def Model_gen(dic, tags=["output_DNN"]):
    models = dic.keys()
    file_names = dic[models[0]].keys()
    for _ in file_names:
        yield _, {tag:{el:imread(dic[el][_][tag]) for el in models} for tag in tags}

def SumImages(list_img):
    res = np.zeros_like(list_img[0])
    for img_bin in list_img:
        res += img_bin
    res = res / len(list_img)
    return res.astype('uint8')

def PostProcessGuess(name, img):
    if "Dist" in name:
        return PostProcess(img, 0, 0.5)
    else:
        img = img.astype('float')
        img = img / 255.
        return PostProcess(img, 14, 0.5)

if __name__ == '__main__':
    list_models = ["UNetHistogramTW2", "UNetDistHistogramTW2", "Test"]
    OUTPUT = join(PATH, "Sum")
    CheckOrCreate(OUTPUT)

    general_dic = GatherMultipleModels(list_models, "Test")
    out_tag = "output_DNN_mean"
    for name, dic in Model_gen(general_dic, tags=[out_tag, "colored_pred", "rgb"]):
        OUTPUT_img = join(OUTPUT, name)
        CheckOrCreate(OUTPUT_img)
        # for el in dic["output_DNN_mean"].keys():
        #     imsave(join(OUTPUT_img, "output_DNN__" + el + ".png"), dic["output_DNN_mean"][el])
        list_pred = []
        for el in dic[out_tag].keys():
            colored_bin = PostProcessGuess(el, dic[out_tag][el])
            colored_bin[colored_bin > 0] = 255
            list_pred.append(colored_bin)
        # pdb.set_trace()
        summed = SumImages(list_pred)
        imsave(join(OUTPUT_img, "summing.png"), summed)

