import pdb
import tensorflow as tf
from os.path import join
from optparse import OptionParser
import numpy as np
from glob import glob
from skimage.io import imread
from ImageTransform import flip_horizontal, flip_vertical
from random_utils import sliding_window, UNetAugment

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

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

def LoadRGB_GT_QUEUE(imgpath, dic, stepSize, windowSize, unet):
    img = imread(imgpath)[:,:,0]
    if unet:
        img = UNetAugment(img)
    label = imread(dic[imgpath])
    for x, y, h, w, sub_lab in sliding_window(label, stepSize, windowSize):
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

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH,
                    BATCH_SIZE, N_THREADS, CHANNELS=3):
    
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
    
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image_f,
                                           target_height=const_IMG_HEIGHT,
                                           target_width=const_IMG_WIDTH)
    
    resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation_f,
                                           target_height=const_MASK_HEIGHT,
                                           target_width=const_MASK_WIDTH)

    images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
                                                 batch_size=BATCH_SIZE,
                                                 capacity=10 + 3 * BATCH_SIZE,
                                                 num_threads=N_THREADS,
                                                 min_after_dequeue=10)
    
    return images, annotations

