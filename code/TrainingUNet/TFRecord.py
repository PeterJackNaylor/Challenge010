from utils.UsefulFunctionsCreateRecord import CreateTFRecord
from utils.ImageTransform import Identity, Flip, Rotation, OutOfFocus, ElasticDeformation, HE_Perturbation, HSV_Perturbation
import numpy as np
from optparse import OptionParser



def ListTransform(n_rot=4, n_elastic=50, n_he=50, n_hsv = 50,
                  var_elast=[1.2, 24. / 512, 0.07], var_hsv=[0.01, 0.07],
                  var_he=[0.07, 0.07]):
    transform_list = [Identity(), Flip(0), Flip(1)]
    if n_rot != 0:
        for rot in np.arange(1, 360, n_rot):
            transform_list.append(Rotation(rot, enlarge=True))

    for sig in [1, 2, 3, 4]:
        transform_list.append(OutOfFocus(sig))

    for i in range(n_elastic):
        transform_list.append(ElasticDeformation(var_elast[0], var_elast[1], var_elast[2]))

    k_h = np.random.normal(1.,var_he[0], n_he)
    k_e = np.random.normal(1.,var_he[1], n_he)

    for i in range(n_he):
        transform_list.append(HE_Perturbation((k_h[i],0), (k_e[i],0), (1, 0)))


    k_s = np.random.normal(1.,var_hsv[0], n_hsv)
    k_v = np.random.normal(1.,var_hsv[1], n_hsv)

    for i in range(n_hsv):
        transform_list.append(HSV_Perturbation((1,0), (k_s[i],0), (k_v[i], 0))) 

    transform_list_test = [Identity()]

    return transform_list, transform_list_test


if __name__ == '__main__':
    print "List transform for data augmentation is not applicable yet"
    parser = OptionParser()
    parser.add_option('--tf_record', dest="TFRecord", type="str")
    parser.add_option('--path', dest="path", type="str")
    parser.add_option('--test', dest="test", type="int")
    parser.add_option('--size_train', dest="size_train", type="int")
    parser.add_option('--split', dest="split", type="str")
    parser.add_option('--unet', dest="unet", type="int")
    parser.add_option('--seed', dest="seed", type="int")
    (options, args) = parser.parse_args()

    OUTNAME = options.TFRecord
    PATH = options.path
    TEST = options.test
    SIZE = options.size_train
    SIZE = (SIZE, SIZE)
    SPLIT = options.split
    var_elast = [1.3, 0.03, 0.15]
    var_he  = [0.01, 0.2]
    var_hsv = [0.2, 0.15]
    UNET = True if options.unet == 1 else False
    SEED = options.seed
    

    t_l, t_l_t = ListTransform(n_elastic=0, 
                               var_elast=var_elast,
                               var_hsv=var_hsv,
                               var_he=var_he) 



    CreateTFRecord(OUTNAME, PATH, TEST, SIZE,
                   t_l, UNET, SEED, SPLIT=SPLIT)
