import time
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.image import extract_patches_2d

# loss functions -------------------------------------------
def cross_entropy(data_dict):
    logits = data_dict['logits']
    labels = data_dict['labels']
    loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    return loss_map

def balance_cross_entropy(data_dict):
    loss_map = cross_entropy(data_dict)
    weight_map = balance_weight_map(data_dict)
    return loss_map * weight_map

def feedback_cross_entropy(data_dict, alpha=3, beta=100):
    loss_map = cross_entropy(data_dict)
    weight_map = feedback_weight_map(data_dict, alpha, beta)
    return loss_map * weight_map

def mse(logits, labels):
    if np.ndim(labels) < 2:
        labels = np.expand_dims(labels, -1) 
    loss_map = tf.reduce_mean(tf.square(logits-labels), -1)
    return loss_map


def dice_coefficient(data_dict):
    logits = tf.cast(data_dict['logits'], tf.float32)
    labels = tf.cast(data_dict['labels'], tf.float32)
    axis = tuple(range(1, len(labels.shape) - 1)) if len(labels.shape) > 2 else -1
    pred = tf.nn.softmax(logits)
    # pred = tf.one_hot(tf.argmax(logits, -1), labels.shape[-1])
    
    intersection = tf.reduce_sum(pred * labels, axis)
    sum_ = tf.reduce_sum(pred + labels, axis)
    dice = 1 - 2 * intersection / sum_
    return dice

def balanced_dice_coefficient(data_dict):
    labels = tf.cast(data_dict['labels'], tf.float32)
    dice_loss = dice_coefficient(data_dict)
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 2 else -1
    c = 1/(np.sum(1/(np.sum(labels, axis=axis))))
    balanced_weight = c/(np.sum(labels, axis=axis))
    dice = dice_loss * balanced_weight
    return dice

# @tf.function
def spatially_constrained_loss(data_dict, kernal_size=3, sigma=0.5):
    # time_start = time.time()
    orgs = tf.cast(data_dict['orgs'], tf.float32)
    logits = tf.cast(data_dict['logits'], tf.float32)

    ndim = len(logits.shape)
    assert ndim in [4, 5], 'only allow 2d or 3d images without RGB channel.'
    if type(kernal_size) is int:
        kernal_size = [1] + [kernal_size,] * (ndim-2) + [1]
    elif type(kernal_size) is list:
        kernal_size = [1] + kernal_size + [1]
    strides = [1,] * ndim
    rates = [1,] * ndim

    probs = tf.nn.softmax(logits)
    confs = tf.reduce_max(probs, -1, keepdims=True)
    arg_preds = tf.cast(tf.expand_dims(tf.argmax(probs, -1), -1), tf.float32)

    if ndim == 4:
        p_zmask = tf.image.extract_patches(tf.ones(confs.shape), kernal_size, strides, rates, padding='SAME')
        p_confs = tf.image.extract_patches(confs, kernal_size, strides, rates, padding='SAME')
        p_orgs = tf.image.extract_patches(orgs, kernal_size, strides, rates, padding='SAME')
        p_preds = tf.image.extract_patches(arg_preds, kernal_size, strides, rates, padding='SAME')
    elif ndim == 5:
        p_zmask = tf.extract_volume_patches(tf.ones(confs.shape), kernal_size, strides, padding='SAME')
        p_confs = tf.extract_volume_patches(confs, kernal_size, strides, padding='SAME')
        p_orgs = tf.extract_volume_patches(orgs, kernal_size, strides, padding='SAME')
        p_preds = tf.extract_volume_patches(arg_preds, kernal_size, strides, padding='SAME')
    # p_zmask_list = tf.unstack(p_zmask, axis=-1)
    # p_zmask_list[np.prod(kernal_size) // 2] = tf.zeros(confs.shape[:-1])
    # p_zmask = tf.stack(p_zmask_list, axis=-1)
    p_exp = tf.exp(-tf.square(orgs - p_orgs) / (2*sigma**2))
    p_exp = p_zmask * p_exp
    p_mask = 2 * tf.cast(arg_preds == p_preds, tf.float32) - 1
    # p_mask = tf.where(arg_preds == p_preds, 1., -1.)
    u_ij = p_exp * p_mask
    P_ij = confs * p_confs
    F_ij = u_ij * P_ij
    F_i = (tf.reduce_sum(F_ij, -1) - tf.reshape(confs**2, confs.shape[:-1])) / (tf.reduce_sum(p_exp, -1) - 1 + 1e-9)
    sc_loss_map = 1 - F_i

    # print('ray time cost: {}'.format(time.time() - time_start))

    return sc_loss_map

# ----------------------------------------------------------

# weight maps ----------------------------------------------

def balance_weight_map(data_dict, epsilon=1e-9):
    labels = data_dict['labels']
    axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 1 else -1
    c = 1/(np.sum(1/(epsilon + np.sum(labels, axis=axis))))
    weight_map = np.sum(labels * np.tile(c/(epsilon + np.sum(labels, axis=axis, keepdims=True)), list(labels.shape[0:-1]) + [1]), axis=-1)
    return weight_map

def feedback_weight_map(data_dict, alpha=3, beta=100):
    logits = data_dict['logits']
    labels = data_dict['labels']
    probs = tf.nn.softmax(logits, -1)
    p = np.sum(probs * labels, axis=-1)
    weight_map = np.exp(-np.power(p, beta)*np.log(alpha))
    return weight_map 

# ----------------------------------------------------------