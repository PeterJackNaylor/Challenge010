from glob import glob
from os.path import join

def GatherFiles(PATH, FOLD_TEST, split="train"):
    folder_train = [el for el in glob(join(PATH, 'Slide_*')) if "Slide_" + str(FOLD_TEST) not in el]
    folder_test = [el for el in glob(join(PATH, 'Slide_*')) if "Slide_" + str(FOLD_TEST) in el]

    train_images = []
    for fold in folder_train:
        train_images += glob(join(fold, '*.png'))
    test_images = []
    for fold in folder_test:
        test_images += glob(join(fold, '*.png'))
    def naming_scheme(name):
        return name.replace('Slide', 'GT').replace('.png', '_mask.png')
    dic_train_gt = {el:naming_scheme(el) for el in train_images}
    dic_test_gt = {el:naming_scheme(el) for el in test_images}
    if split == "train":
        return train_images, dic_train_gt
    else:
        return test_images, dic_test_gt
