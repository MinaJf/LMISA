from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras.layers import *
import nibabel as nib
import os, sys, re
from utils import process_methods as PM
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def preprocessing( x_i, x_l ):
        x_i = x_i.reshape((1, 64, 128, 25, 1))
        x_l = x_l.reshape((1, 64, 128, 25))
        x = PM.zero_mean(x_i)

        x_l_p = PM.one_hot(x_l, [0, 255])
        y = PM.channel_check(x_l_p, 2).astype(np.float32)

        # x = tf.constant(x_i_p, tf.dtypes.float32)
        # y = tf.constant(x_l_p, tf.dtypes.float32)

        return x, y
def _read_save_tf(path, filename):
    # path to save the TFRecords file

    # open the file
    writer = tf.io.TFRecordWriter(filename)
    filename_pairs = gen_filename_pairs(path, '_img.nii.gz', '_lab.nii.gz')
    for v_filename, l_filename in filename_pairs:
        print("Processing:")
        print("  volume: ", v_filename)
        print("  label:  ", l_filename)

        # The volume, in nifti format
        v_nii = nib.load(v_filename)
        # The volume, in numpy format
        v_np = v_nii.get_data().astype(np.float32)
        # v_raw =  v_np.tostring()

        # The label, in nifti format
        l_nii = nib.load(l_filename)
        # The label, in numpy format
        l_np = l_nii.get_data().astype(np.float32)
        # l_raw = l_np.tostring()
        d_ct, d_mri = preprocessing(v_np, l_np)
        feature = {'img': _float_feature(d_ct.ravel()),
                   'lab': _float_feature(d_mri.ravel()),
                   }

        # Create an example protocol buffer
        example = tf.train.Example(features=tf.train.Features(feature=feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()

def decode(serialized_example):
    # raw_size = [128, 128, 64]
    feature_description = {
        'img': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing = True),
        'lab': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }

    data = tf.io.parse_single_example(serialized_example, feature_description)
    data['img'] = tf.reshape(data['img'], ( 64, 128, 25, 1))
    data['lab'] = tf.reshape(data['lab'], ( 64, 128, 25, 2))


    # for i in data:
    #     xs = i['img'].numpy().reshape(( 128, 128, 64, 1))
    #     ys = i['lab'].numpy().reshape(( 128, 128, 64))
    #
    #     xs = PM.channel_check(xs, 1)
    #     xs = PM.zero_mean(xs)
    #
    #     ys = PM.channel_check(ys, 5)

    return data

def listfiles(folder):
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            yield os.path.join(root, filename)

def gen_filename_pairs(data_dir, v_re, l_re):
    unfiltered_filelist=list(listfiles(data_dir))
    input_list = [item for item in unfiltered_filelist if re.search(v_re,item)]
    label_list = [item for item in unfiltered_filelist if re.search(l_re,item)]
    print("input_list size:    ", len(input_list))
    print("label_list size:    ", len(label_list))
    if len(input_list) != len(label_list):
        print("input_list size and label_list size don't match")
        raise Exception
    return zip(input_list, label_list)