import os, sys

from skimage.measure import label
import skimage.io
from skimage.feature import register_translation

import EdgeClassification
import pdb
import shutil

from skimage.transform import rescale, resize, downscale_local_mean

in_folder = '/Users/naylorpeter/Desktop/NucleiKaggle/results/'
original_in_folder = '/Users/naylorpeter/Desktop/NucleiKaggle/results'

retriever = EdgeClassification.ImageRetriever(in_folder, nn_names=['rcnn_external'])
original_retriever = EdgeClassification.ImageRetriever(original_in_folder, nn_names=['ContrastUNet'])

def info():
    image_names = retriever.get_training_image_names()
    orig_image_names = original_retriever.get_training_image_names()

    for img_name, orig_name in zip(image_names, orig_image_names):
        imin = retriever.get_original_image(img_name)
        imorig = original_retriever.get_original_image(img_name)

        out_folder = retriever.get_image_folder(img_name, nn_name='rcnn_external')
        prediction = skimage.io.imread(os.path.join(out_folder, 'pred.png'))

        if (imin.shape[0] / 2 == imorig.shape[0]) and (imin.shape[1] / 2 == imorig.shape[1]):
            im_rescaled = rescale(prediction, 0.5, order=0)
            print '%s:\t(%i,%i)\t(%i,%i)' % (img_name, imorig.shape[0], imorig.shape[1], im_rescaled.shape[0], im_rescaled.shape[1])

            try:
                os.remove(os.path.join(out_folder, 'output_DNN.png'))
                skimage.io.imsave(os.path.join(out_folder, 'output_DNN.png'), im_rescaled)
            except:
                pdb.set_trace()
        else:
            im_rescaled = rescale(prediction, 0.5, order=0)
            orig_width = imorig.shape[1]
            orig_height = imorig.shape[0]
            delta_x = im_rescaled.shape[1] - imorig.shape[1]
            delta_y = im_rescaled.shape[1] - imorig.shape[1]
            offset_x = delta_x / 2
            offset_y = delta_y / 2
            im_rescaled = im_rescaled[offset_y:(offset_y+orig_height), offset_x:(offset_x+orig_width), :]
            print '%s:\t(%i,%i)\t(%i,%i)' % (img_name, imorig.shape[0], imorig.shape[1], im_rescaled.shape[0], im_rescaled.shape[1])
            try:
                os.remove(os.path.join(out_folder, 'output_DNN.png'))
                skimage.io.imsave(os.path.join(out_folder, 'output_DNN.png'), im_rescaled)
            except:
                pdb.set_trace()

            #shift, error, diffphase = register_translation(imorig, imin)


info() 