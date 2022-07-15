import os
import sys
import shutil
import argparse
import numpy as np
import tensorflow as tf
import scipy.io as sio
import glob
import nibabel as nib
# sys.path.append('git/_framework')
from utils import util as U
from core.data_provider import DataProvider
from core.trainer_tf_uni import Trainer
from core.data_processor import SimpleImageProcessor
from core.learning_rate import StepDecayLearningRate
from model.model_wgan_uni_3D import GANModel
from model.gan_10k_3D import Generator, Discriminator
import timeit
tf.config.experimental_run_functions_eagerly(True)
start = timeit.default_timer()
epochs = 1
batch_size = 2
mini_batch_size = 1
eval_batch_size = 1
eval_frequency = 2
use_bn = False
use_res = True
learning_rate = 0.001
g_alpha = 40
g_lambda = 10
g_beta = 10
dropout = 0
resize = None
output_path = 'agecls/wgan_10k_640_zm_mm'
output_path = '{}_batch{}_bn{}_dp{}_a{}_la{}_lr{}'.format(output_path, batch_size, int(use_bn), dropout, g_alpha,
                                                          g_lambda, learning_rate)
saved_filelists = 'std_sex_640_160_62.mat'

import platform

if platform.system() == 'Windows':
    data_path = './'
    output_path = 'results/' + output_path
if platform.system() == 'Linux':
    data_path = './'
    output_path = 'results/' + output_path

# if os.path.exists(output_path):
#     shutil.rmtree(output_path, ignore_errors=True)

org_path = data_path + '/std96'
saved_filelists = data_path + '/' + saved_filelists
# load filenames
# file_mat = sio.loadmat(saved_filelists)
# load filenames
data_path = '.'
train_list_CT = glob.glob(data_path + '/trainCT/*_img.nii.gz')
valid_list_CT = glob.glob(data_path + '/validationCT/*_img.nii.gz')
train_list_MRI = glob.glob(data_path + '/trainMRI/*_img.nii.gz')
valid_list_MRI = glob.glob(data_path + '/validationMRI/*_img.nii.gz')
test_list = glob.glob(data_path + '/test/*_img.nii.gz')

# set key for input images and labels
org_suffix = '_img.nii.gz'
lab_suffix = '_lab.nii.gz'

pre = {org_suffix: [ 'zero-mean', ('channelcheck', 1)],
       lab_suffix: [('one-hot', [0, 205, 420, 500, 820]), ('channelcheck', 5)]
       }

pre_ = {org_suffix: [ 'zero-mean', ('channelcheck', 1)],
       lab_suffix: [('one-hot', [0, 205, 420, 500, 820]), ('channelcheck', 5)]
       }
processor = SimpleImageProcessor(pre=pre)
processor_no_p = SimpleImageProcessor(pre=pre_)
train_provider_CT_t = DataProvider(train_list_CT, [org_suffix, lab_suffix],
                                   is_pre_load=False,
                                   is_shuffle=True,
                                   # temp_dir=output_path,
                                   processor=processor)

valid_provider_CT_t = DataProvider(valid_list_CT, [org_suffix, lab_suffix],
                                   is_pre_load=False,
                                   # temp_dir=output_path,
                                   processor=processor)

train_provider_CT = DataProvider(train_list_CT, [org_suffix],
                                 is_pre_load=False,
                                 is_shuffle=True,
                                 # temp_dir=output_path,
                                 processor=processor)

valid_provider_CT = DataProvider(valid_list_CT, [org_suffix],
                                 is_pre_load=False,
                                 # temp_dir=output_path,
                                 processor=processor)

train_provider_MRI = DataProvider(train_list_MRI, [org_suffix],
                                  is_pre_load=False,
                                  is_shuffle=True,
                                  # temp_dir=output_path,
                                  processor=processor)

valid_provider_MRI = DataProvider(valid_list_MRI, [org_suffix],
                                  is_pre_load=False,
                                  # temp_dir=output_path,
                                  processor=processor)

train_provider_CT_t_w = DataProvider(train_list_CT, [org_suffix, lab_suffix],
                                   is_pre_load=False,
                                   is_shuffle=True,
                                   # temp_dir=output_path,
                                   processor=processor_no_p,
                                is_aug=True)

train_provider_CT_w = DataProvider(train_list_CT, [org_suffix],
                                 is_pre_load=False,
                                 is_shuffle=True,
                                 # temp_dir=output_path,
                                 processor=processor_no_p,
                                   is_aug=True)

train_provider_MRI_w = DataProvider(train_list_MRI, [org_suffix],
                                  is_pre_load=False,
                                  is_shuffle=True,
                                  # temp_dir=output_path,
                                  processor=processor_no_p,
                                    is_aug=True)

##################test#######################################
# train_provider_CT_t = DataProvider(train_list_CT, [org_suffix, lab_suffix],
#                                    is_pre_load=False,
#                                    is_shuffle=False,
#                                    # temp_dir=output_path,
#                                    processor=processor)
# for i in range(train_provider_CT_t.size):
#     if i == 82:
#         print()
#     fname = train_provider_CT_t._file_list[train_provider_CT_t._cur_i]
#     data = train_provider_CT_t(1)
#     lab = data[lab_suffix]
#     print('{}: {}  {}'.format(i, np.unique(lab), fname))
# print()
##################################################################
# build model
gen = Generator(2, 4, 16, use_bn=use_bn)
disc = Discriminator(1, 4, 16, use_bn=use_bn)

model = GANModel([gen, disc], org_suffix, lab_suffix, g_alpha=g_alpha, g_lambda=g_lambda, g_beta=g_beta,
                 dropout=0)
gen_lr = StepDecayLearningRate(learning_rate=learning_rate,
                               decay_step=10,
                               decay_rate=0.8,
                               data_size=train_provider_CT.size + train_provider_MRI.size,
                               batch_size=batch_size)
disc_lr = StepDecayLearningRate(learning_rate=learning_rate,
                                 decay_step=10,
                                 decay_rate=0.8,
                                 data_size=train_provider_CT.size + train_provider_MRI.size,
                                 batch_size=batch_size)


gen_optimizer = tf.keras.optimizers.Adam(gen_lr)
disc_optimizer = tf.keras.optimizers.Adam(disc_lr)

trainer = Trainer(model)
# x = train_provider_MRI_w(1)
# gen(x, 0, False)
# disc(x, 0, False)


# train
results = trainer.train(train_provider_CT, train_provider_MRI, valid_provider_CT, valid_provider_MRI,
                        train_provider_CT_t, valid_provider_CT_t, train_provider_CT_w, train_provider_MRI_w,train_provider_CT_t_w,
                        epochs=epochs,
                        batch_size=batch_size,
                        mini_batch_size=mini_batch_size,
                        output_path=output_path,
                        optimizer=[gen_optimizer, disc_optimizer],
                        learning_rate=[gen_lr, disc_lr],
                        eval_frequency=eval_frequency)

# eval
test_provider = DataProvider(test_list, [org_suffix, lab_suffix],
                             is_pre_load=False,
                             processor=processor)



trainer.restore(output_path + '/ckpt/best')
eval_dcit = trainer.eval_test(test_provider, batch_size=eval_batch_size)
with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('Best :' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/best_results.mat', eval_dcit)
#
trainer.restore(output_path + '/ckpt/final')
eval_dcit = trainer.eval_test(test_provider, batch_size=eval_batch_size)
with open(output_path + '/test_eval.txt', 'a+') as f:
    f.write('final:' + U.dict_to_str(eval_dcit) + '\n')
sio.savemat(output_path + '/final_results.mat', eval_dcit)
# print()
stop = timeit.default_timer()

print('Time: ', stop - start)
#trainer.restore(output_path + '/ckpt/final')
# data_dict = test_provider(50)


from PIL import Image

# save predictions
savepath = output_path + '/predictions/'
# evaluate test data

num = 14
img_arr = [[] for i in range(num)]
org_name = [[] for i in range(num)]
test_provider._cur_i = 0
for i in range(num):
    data = test_provider(1)
    idx = i
    img_arr[idx].append(data[org_suffix])
    org_name[idx].append(os.path.split(test_list[i])[-1])
    eval_dcit = trainer.eval_test(data, batch_size=eval_batch_size)
    #dice = np.array(eval_dcit['dice'])
    # print(org_name[idx])
    # print(eval_dcit['dice'])

import nibabel as nib
temp_img = nib.load('./test/8_lab.nii.gz')
temp_affine = temp_img.affine
z = [[] for i in range(num)]
for k in range(len(img_arr)):
    if len(img_arr[k]) <= 0:
        continue
    for img in img_arr[k]:
        [_, out] = gen(img)
        # org_name = os.path.split(test_list[k])[-1]
        img = np.argmax(out, -1)
        org_n = org_name[k]
        # img = o1
               #img = predictions[k,...,1]
        pred = np.squeeze(img)
        # pred[pred==1]=205
        # pred[pred==2]=420
        # pred[pred==3]=500
        # pred[pred==4]=820

        img = nib.Nifti1Image(pred.astype(np.float32), affine=temp_affine)
        nib.save(img, os.path.join(savepath + org_n[0]))