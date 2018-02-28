import pdb
import tensorflow as tf
from os.path import join
from optparse import OptionParser
import numpy as np
from glob import glob
from skimage.io import imread
from utils.ImageTransform import flip_horizontal, flip_vertical
from utils.random_utils import sliding_window, UNetAugment
import math
import scipy.stats as st
from FileCollector import GatherFiles


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def LoadRGB_GT_QUEUE(imgpath, dic, stepSize, windowSize, unet):
    img = imread(imgpath)[:,:,0:3]
    if unet:
        img = UNetAugment(img)
    label = imread(dic[imgpath])
    for x, y, h, w, sub_lab in sliding_window(label, (stepSize, stepSize), windowSize):
        if unet:
            x_u, y_u = x + 92, y + 92
            sub_img = img[(x_u - 92):(x_u + h - x + 92),(y_u - 92):(y_u + w - y + 92)]
        else:
            sub_img = img[x:h, y:w]
        yield sub_img, sub_lab


def CreateTFRecord(OUTNAME, PATH, FOLD_TEST, SIZE,
                   TRANSFORM_LIST, UNET, SEED,
                   SPLIT="train"):

    tfrecords_filename = OUTNAME
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    images, dic_gt = GatherFiles(PATH, FOLD_TEST, SPLIT)
    original_images = []
    n_samples = 0
    for img_path in images:
        for img, annotation in LoadRGB_GT_QUEUE(img_path, dic_gt, SIZE[0], SIZE, UNET):
            n_samples += 1
            height_img = img.shape[0]
            width_img = img.shape[1]

            height_mask = annotation.shape[0]
            width_mask = annotation.shape[1]
          
            original_images.append((img, annotation))
              
            img_raw = img.tostring()
            annotation_raw = annotation.tostring()
              
            example = tf.train.Example(features=tf.train.Features(feature={
                  'height_img': _int64_feature(height_img),
                  'width_img': _int64_feature(width_img),
                  'height_mask': _int64_feature(height_mask),
                  'width_mask': _int64_feature(width_mask),
                  'image_raw': _bytes_feature(img_raw),
                  'mask_raw': _bytes_feature(annotation_raw)}))
              
            writer.write(example.SerializeToString())

    print "I have written {} images in this record".format(n_samples)
    writer.close()

def coin_flip(p):
    with tf.name_scope("coin_flip"):
        return tf.reshape(tf.less(tf.random_uniform([1],0, 1.0), p), [])

def return_shape(img):
    shape = tf.shape(img)
    x = shape[0]
    y = shape[1]
    z = shape[2]
    return x, y, z

def expend(img, marge):
    with tf.name_scope("expend"):    
        border_left = tf.image.flip_left_right(img[:, :marge])
        border_right= tf.image.flip_left_right(img[:, -marge:])
        width_ok = tf.concat([border_left, img, border_right], 1)
        height, width, depth = return_shape(width_ok)
        border_low = tf.image.flip_up_down(width_ok[:marge, :])
        border_up  = tf.image.flip_up_down(width_ok[-marge:, :])
        final_img = tf.concat([border_low, width_ok, border_up], 0)
        return final_img
def rotate_rgb(img, angles, marge):
    with tf.name_scope("rotate_rgb"):
        ext_img = expend(img, marge)
        height, width, depth = return_shape(ext_img)
        rot_img = tf.contrib.image.rotate(ext_img, angles, interpolation='BILINEAR')
        reslice = rot_img[marge:-marge, marge:-marge]
        return reslice
def rotate_label(img, angles, marge):
    with tf.name_scope("rotate_lbl"):
        ext_img = expend(img, marge)
        height, width, depth = return_shape(ext_img)
        rot_img = tf.contrib.image.rotate(ext_img, angles, interpolation='NEAREST')
        reslice = rot_img[marge:-marge, marge:-marge]
        return reslice
def flip_left_right(ss, coin):
    with tf.name_scope("flip_left_right"):   
        return tf.cond(coin,
                    lambda: tf.image.flip_left_right(ss),
                    lambda: ss)
def flip_up_down(ss, coin):
    with tf.name_scope("flip_up_down"):   
        return tf.cond(coin,
                    lambda: tf.image.flip_up_down(ss),
                    lambda: ss)
def RandomFlip(image, label, p):
    with tf.name_scope("RandomFlip"):
        coin_up_down = coin_flip(p)
        image = flip_up_down(image, coin_up_down)
        label = flip_up_down(label, coin_up_down)
        coin_left_right = coin_flip(p)
        image = flip_left_right(image, coin_left_right)
        label = flip_left_right(label, coin_left_right)
        return image, label

def RandomRotate(image, label, p):
    with tf.name_scope("RandomRotate"):
        coin_rotate = coin_flip(p)
        angle_rad = 0.5 * math.pi
        angles = tf.random_uniform([1], 0, angle_rad)
    #    h_i, w_i, d_i = return_shape(image)
        h_i, w_i, d_i = 396, 396, 3
        marge_rgb = int(max(h_i, w_i) * (2 - 1.414213) / 1.414213)
        image = tf.cond(coin_rotate, lambda: rotate_rgb(image, angles, marge_rgb)
                                   , lambda: image)
    #    h_l, w_l, d_l = return_shape(label)
        h_l, w_l, d_l = 212, 212, 1
        marge_l = int(max(h_l, w_l) * (2 - 1.414213) / 1.414213)
        label = tf.cond(coin_rotate, lambda: rotate_label(label, angles, marge_l)
                                   , lambda: label)
        return image, label

def GaussianKernel(kernlen=21, nsig=3, channels=1, sigma=1.):
    interval = (2 * nsig + 1.)/(kernlen)
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x, scale=sigma))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype = np.float32)
    out_filter = out_filter.reshape((kernlen, kernlen, 1, 1))
    kernel_mat = np.repeat(out_filter, channels, axis = 2)
    var = tf.constant(kernel_mat.flatten(), shape=kernel_mat.shape,dtype=tf.float32)
    return var

def Blur(image, sigma, channels):
    with tf.name_scope("Blur"):
        K = GaussianKernel(nsig=3, channels=channels, sigma=sigma)
        height, width, depth = return_shape(image)
        image = tf.expand_dims(image, axis=0)
        image = tf.nn.depthwise_conv2d(image, K, strides=[1,1,1,1], padding='SAME')
        out = tf.squeeze(image)
        out = tf.reshape(image, shape=[height, width, 3])
        return out
def RandomBlur(image, label, p):
    with tf.name_scope("RandomBlur"):
        coin_blur = coin_flip(p)
        start, end = 0, 3
        sigma_list = [0.1,0.2,0.3]
        elems = tf.convert_to_tensor(sigma_list)
        samples = tf.multinomial(tf.log([[0.60, 0.30, 0.10]]), 1)
        nsig = tf.cast(samples[0][0], tf.int32)
        sigma = elems[nsig]
        for i in range(start, end):
            sigma_try = tf.constant([i])
            check = tf.reshape(tf.equal(nsig, sigma_try), [])
            apply_f = tf.logical_and(coin_blur,check)
            # apply_f = tf.reshape(apply_f, [])
            image = tf.cond(apply_f, lambda: Blur(image, sigma_list[i], channels=3)
                                   , lambda: image)
        return image, label

def ChangeBrightness(image, delta):
    with tf.name_scope("ChangeBrightness"):
        hsv = tf.image.rgb_to_hsv(image)
        first_channel = hsv[:,:,0:2]
        hsv2 = hsv[:,:,2]
        num = tf.multiply(delta, 255.)
        hsv2_aug = tf.clip_by_value(hsv2 + num, 0., 255.)
        hsv2_exp = tf.expand_dims(hsv2_aug, 2)
        hsv_new = tf.concat([first_channel, hsv2_exp], 2)

        return tf.image.hsv_to_rgb(hsv_new)

def RandomBrightness(image, label, p):
    with tf.name_scope("RandomBrightness"):
        coin_hue = coin_flip(p)
        delta = tf.truncated_normal([1], mean=0.0, stddev=0.1,
                                  dtype=tf.float32)
        image = tf.cond(coin_hue, lambda: ChangeBrightness(image, delta)
                                , lambda: image)
        return image, label

def ElasticDeformation(image, alpha, alpha_affine, sigma):
    with tf.name_scope("ElasticDeformation"):
        return 'hi'


def Generate(img, alpha_affine):
    x, y, z = return_shape(img)
    cent = tf.stack([x, y]) / 2
    sq_size = tf.minimum(x, y) / 3
#    return sq_size
    second = tf.stack([x / 2 + sq_size, y / 2 - sq_size])
#    return cent + sq_size, second, cent - sq_size
    pts1 = tf.cast(tf.stack([cent + sq_size, second
                                   , cent - sq_size]), tf.float32)
    pts2 = pts1 + tf.reshape(tf.random_uniform([6], -alpha_affine, alpha_affine), [3,2])

    return pts1, pts2
#   return M
def RandomElasticDeformation(image, annotation, p,
                             alpha=1, 
                             alpha_affine=1,
                             sigma=1):
    with tf.name_scope("RandomElasticDeformation"):
        coin_ela = coin_flip(p)
        annotation = expend(annotation, 92)
        x, y, z = return_shape(annotation)
        alpha = tf.multiply(x, alpha)
        alpha_affine = tf.multiply(x, alpha_affine)
        sigma = tf.multiply(x, sigma)
        M = Genrate(img, alpha_affine)

        image, annotation = tl.prepro.elastic_transform_multi([image, annotation],
                            alpha=720, sigma=24, is_random=True)
        return image, annotation[92:-92,92:-92]

def augment(image_f, annotation_f):
    with tf.name_scope("DataAugmentation"):
        image_f, annotation_f = RandomFlip(image_f, annotation_f, 0.5)    
        image_f, annotation_f = RandomRotate(image_f, annotation_f, 0.2)  
        image_f, annotation_f = RandomBlur(image_f, annotation_f, 0.2)
        image_f, annotation_f = RandomBrightness(image_f, annotation_f, 0.2)
        return image_f, annotation_f

def augment_npy(image, annotation):
    print "not implemented"

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH,
                    BATCH_SIZE, N_THREADS, CHANNELS=3):
    
    with tf.name_scope("LoadingImagesIntoNet"):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height_img': tf.FixedLenFeature([], tf.int64),
            'width_img': tf.FixedLenFeature([], tf.int64),
            'height_mask': tf.FixedLenFeature([], tf.int64),
            'width_mask': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        height_img = tf.cast(features['height_img'], tf.int32)
        width_img = tf.cast(features['width_img'], tf.int32)

        height_mask = tf.cast(features['height_mask'], tf.int32)
        width_mask = tf.cast(features['width_mask'], tf.int32)

        const_IMG_HEIGHT = IMAGE_HEIGHT + 184
        const_IMG_WIDTH = IMAGE_WIDTH + 184

        const_MASK_HEIGHT = IMAGE_HEIGHT
        const_MASK_WIDTH = IMAGE_WIDTH


        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
        
        
        image_shape = tf.stack([height_img, width_img, CHANNELS])
        annotation_shape = tf.stack([height_mask, width_mask, 1])
        
        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)
        image_size_const = tf.constant((const_IMG_HEIGHT, const_IMG_WIDTH, CHANNELS), dtype=tf.int32)
        annotation_size_const = tf.constant((const_MASK_HEIGHT, const_MASK_WIDTH, 1), dtype=tf.int32)
        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.
        image_f = tf.cast(image, tf.float32)
        annotation_f = tf.cast(annotation, tf.float32)
        image_f1, annotation_f1 = augment(image_f, annotation_f)

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image_f1,
                                               target_height=const_IMG_HEIGHT,
                                               target_width=const_IMG_WIDTH)
        
        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation_f1,
                                               target_height=const_MASK_HEIGHT,
                                               target_width=const_MASK_WIDTH)
        images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                     batch_size=BATCH_SIZE,
                                                     capacity=100 + 3 * BATCH_SIZE,
                                                     num_threads=N_THREADS,
                                                     min_after_dequeue=100)
        
        return images, annotations

if __name__ == '__main__':
    from skimage.io import imread
    path_rgb = "/Users/naylorpeter/Desktop/NucleiKaggle/dataset/stage1_train/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552/images/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552.png"
    path_rgb = "/Users/naylorpeter/Desktop/NucleiKaggle/dataset/stage1_train/0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed/images/0121d6759c5adb290c8e828fc882f37dfaf3663ec885c663859948c154a443ed.png"
    img = imread(path_rgb)[:,:,0:3]
    path_lbl = "/Users/naylorpeter/Desktop/NucleiKaggle/code/Data/work/35/9dbcf463fbddad9c0e0f86dc43000d/00071198d059ba7f5914a526d124d28e6d010c92466da21d4a04cd5413362552_mask.png"
    lab = imread(path_lbl)[:,:,np.newaxis]
    RGB = tf.placeholder(tf.float32, shape=(None, None, None))
    LAB = tf.placeholder(tf.int32, shape=(None, None, None))
    xRGB = expend(RGB, 20)
    xLAB = expend(LAB, 20)
    rLAB = rotate_label(LAB, 45., 60)
    rRGB = rotate_rgb(RGB, 45., 60)
    BlurRGB, BlurLab = RandomBlur(RGB, LAB, 1)
    coin_blur = coin_flip(1.)
    sess = tf.Session()
    elems = tf.convert_to_tensor([1,2,3,4])
    samples = tf.multinomial(tf.log([[0.99, 0.001, 0.001, 0.008]]), 1)
    sigma = elems[tf.cast(samples[0][0], tf.int32)]
    sigma_try = tf.constant([1])
    check = tf.reshape(tf.equal(sigma, sigma_try), [])

    apply_f = tf.logical_and(coin_blur,check)
    # for i in range(start, end):
    #     sigma_try = tf.constant([i])
    #     apply_f = tf.logical_and(coin_blur, tf.equal(sigma, sigma_try))
    #     apply_f = tf.reshape(apply_f, [])
    # Evaluate the tensor `c`.
#    r_RGB, r_LAB = since
    # HUE_RGB = ChangeBrightness(RGB, 0.)
    # HUEminus_RGB = ChangeBrightness(RGB, -0.5)
    # HUEpositiv_RGB = ChangeBrightness(RGB, 0.5)
    # RANDOM_BRIG, LAB1 = RandomBrightness(RGB, LAB, 1.)
    # RANDOM_BRIG2, LAB2 = RandomBrightness(RGB, LAB, 1.)
    # RANDOM_BRIG3, LAB3 = RandomBrightness(RGB, LAB, 0.)
    M = Generate(img, 1)
    #rgb = ElasticDeformation(RGB, 1, 1, 1)
    #rgb_p, lal_p = sess.run([rgb, lab], feed_dict={RGB:img, LAB:lab})
    print sess.run(M, feed_dict={RGB:img})

