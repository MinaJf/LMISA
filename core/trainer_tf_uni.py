import os
import time
import shutil
import numpy as np
import tensorflow as tf
from utils import util_latent as UL
from utils import util as U
from utils.eval_saver import save_str, save_img
import nibabel as nib
import scipy

from scipy import ndimage, misc
# from augment3D import Augment3D

from volumentations import *
from utils.process_methods import one_hot, min_max, channel_check, zero_mean
import nibabel as nib
class Trainer:
    def __init__(self, model):
        self.model = model

    # @tf.function
    def train(self,
              train_provider_CT,
              train_provider_MRI,
              validation_provider_CT,
              validation_provider_MRI,
              train_provider_CT_t,
              validation_provider_CT_t,
              train_provider_CT_w,
              train_provider_MRI_w,
              train_provider_CT_t_w,
              epochs,
              batch_size,
              output_path,
              optimizer=None,
              learning_rate=None,
              mini_batch_size=None,
              eval_frequency=1,
              is_save_train_imgs=False,
              is_save_valid_imgs=True,
              strategy=None):

        self.strategy = strategy

        if learning_rate is None:
            learning_rate = optimizer.learning_rate.numpy()
        if type(learning_rate) is not list:
            learning_rate = [learning_rate]
        iters = (train_provider_CT.size + train_provider_MRI.size) / (batch_size*2)
        # iters = 160
        assert iters > 0 and iters % 1 == 0, 'batch size {} does not match the data size {}.'.format(batch_size,
                                                                                                     (
                                                                                                     train_provider_CT.size + train_provider_MRI.size))
        mini_batch_size = batch_size if mini_batch_size is None else mini_batch_size
        mini_iters = batch_size / mini_batch_size
        assert mini_iters > 0 and mini_iters % 1 == 0, 'mini batch size {} does not match the batch size {}.'.format(
            mini_batch_size, batch_size)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        net_args = {}
        for i in range(len(nets)):
            net_args.update({'net%d' % i: nets[i]})
        ckpt = tf.train.Checkpoint(**net_args)

        print(
            'Start training: epochs {}, learning rate {}, batch size {}, mini-batch size {}, training data {}, validation data {}.'
            .format(epochs, [str(lr) for lr in learning_rate], batch_size, mini_batch_size, (train_provider_CT.size + train_provider_MRI.size) ,(validation_provider_CT.size+ validation_provider_MRI.size)))

        train_eval_str = {}
        valid_eval_str = {}
        best_loss = float('inf')
        time_start = time.time()
        org_suffix = '_img.nii.gz'
        lab_suffix = '_lab.nii.gz'
        #with strategy.scope():
        for ep in range(epochs):
            ep_time_start = time.time()
            for _ in range(int(iters)):
                grads = None
                for _ in range(int(mini_iters)):

                    feed_dict_CT = train_provider_CT_w(mini_batch_size)
                    feed_dict_MRI = train_provider_MRI_w(mini_batch_size)
                    feed_dict_CT_lab = train_provider_CT_t_w(mini_batch_size)

                    # savepath = 'D:/TensorFlow/WGAN_latent/test/'
                    #
                    # lab = feed_dict_CT_lab['_lab.nii.gz']
                    # img = feed_dict_CT_lab['_img.nii.gz']
                    #
                    # temp_img = nib.load('./test/19augr5_img.nii.gz')
                    # temp_affine = temp_img.affine
                    #
                    # img = img[0, ..., 0]
                    # lab = np.argmax(lab, -1)
                    # lab = lab[0, ...]
                    # img = nib.Nifti1Image(img.astype(np.float32), affine=temp_affine)
                    # nib.save(img, os.path.join(savepath + 'img'))
                    #
                    # lab = nib.Nifti1Image(lab.astype(np.float32), affine=temp_affine)
                    # nib.save(lab, os.path.join(savepath + 'lab'))

                    mini_grads = self.model.get_grads(feed_dict_CT, feed_dict_MRI, feed_dict_CT_lab)
                    grads = self._grads_add(grads, mini_grads)
                grads = self._grads_div(grads, mini_iters)
                if type(grads) is tuple:
                    assert len(optimizer) == len(grads), 'Number of optimizer should equal to number of networks.'
                    for gi in range(len(grads)):
                        optimizer[gi].apply_gradients(zip(grads[gi], self.model.net[gi].trainable_variables))
                        # optimizer[gi] = strategy.experimental_run_v2(apply_gradients,(zip(grads[gi], self.model.net[gi].trainable_variables,)))
                else:
                    optimizer.apply_gradients(zip(grads, self.model.net.trainable_variables))
            ep_train_time = time.time() - ep_time_start
            ep_eval_time = 0
            if ep % eval_frequency == 0 or ep == epochs - 1:
                ep_train_eval = self.eval(train_provider_CT_w, train_provider_MRI_w ,train_provider_CT_t_w, batch_size=mini_batch_size, print_str=False,
                                          need_imgs=is_save_train_imgs)
                ep_valid_eval = self.eval(validation_provider_CT, validation_provider_MRI, validation_provider_CT_t, batch_size=mini_batch_size, print_str=False,
                                          need_imgs=is_save_valid_imgs)
                ep_eval_time = time.time() - ep_train_time - ep_time_start
                if is_save_train_imgs:
                    save_img(ep_train_eval[1], '{}/train_imgs/'.format(output_path), ep)
                    ep_train_eval = ep_train_eval[0]
                if is_save_valid_imgs:
                    save_img(ep_valid_eval[1], '{}/valid_imgs1/'.format(output_path), ep)
                    save_img(ep_valid_eval[2], '{}/valid_imgs2/'.format(output_path), ep)
                    ep_valid_eval = ep_valid_eval[0]
                save_str(ep_train_eval, '{}/train_eval.txt'.format(output_path), ep)
                save_str(ep_valid_eval, '{}/valid_eval.txt'.format(output_path), ep)
                # save best ckpt
                if np.mean(ep_valid_eval['loss']) < best_loss:
                    ckpt.write(output_path + '/ckpt/best')
                    best_loss = np.mean(ep_valid_eval['loss'])

                    # time_ep_save_imgs_end = time.time()
            train_log = (
            'epoch {} ------ time cost: overall {:.1f} ------ step training {:.1f} ------ step evaluation {:.1f} ------ learning rate: {} ------'
            .format(ep, time.time() - time_start, ep_train_time, ep_eval_time, [str(lr) for lr in learning_rate]))

            if ep % eval_frequency == 0 or ep == epochs - 1:
                train_log += ('\n  train      : {}'.format(U.dict_to_str(ep_train_eval)) + \
                              '\n  validation : {}'.format(U.dict_to_str(ep_valid_eval)))

            print(train_log)
            with open(output_path + '/train_log.txt', 'a+') as f:
                f.write(train_log + '\n')

            train_eval_str = U.dict_append(train_eval_str, ep_train_eval)
            valid_eval_str = U.dict_append(valid_eval_str, ep_valid_eval)

            # TODO add early stopping and best ckpt save
            # TODO add tensorboard summary
            ckpt.write(output_path + '/ckpt/final')

        return train_eval_str, valid_eval_str

    def get_augmentation(self, patch_size):
        return Compose([
            Rotate((0, 0), (0, 0), (0, 15), p=0.5),
        ], p=1.0)

    def data_augmentation2_org(self, z):
        org_suffix = 'image'
        img_tr = None
        image = z[org_suffix]
        width = 64
        height = 128
        all_axes = [(1, 0), (1, 2), (0, 2)]
        angle = np.random.randint(low=5, high=15 + 1)
        axes_random_id = np.random.randint(low=0, high=len(all_axes))
        axes = all_axes[axes_random_id]
        img = scipy.ndimage.rotate(image, angle, axes=axes, reshape=False)
        # for i in range(0, image.shape[2]):
        #     tmp_img_tr = cv2.resize(img [..., i], (width, height), interpolation=cv2.INTER_CUBIC)
        #     tmp_img_tr = np.expand_dims(tmp_img_tr, -1)
        #     img_tr = tmp_img_tr if img_tr is None else np.concatenate((img_tr, tmp_img_tr), -1)


        z[org_suffix] = img

        return  z

    def data_augmentation2_lab(self, z):
        org_suffix = 'image'
        lab_suffix = 'mask'

        imag = z[lab_suffix]
        mask = imag
        img_tr = None
        m1 = None
        imag = z[org_suffix]
        image = imag
        width = 64
        height = 128
        all_axes = [(1, 0), (1, 2), (0, 2)]
        angle = np.random.randint(low=5, high=15 + 1)
        axes_random_id = np.random.randint(low=0, high=len(all_axes))
        axes = all_axes[axes_random_id]
        img = scipy.ndimage.rotate(image, angle, axes=axes, reshape=False)
        msk  = scipy.ndimage.rotate(mask, angle, axes=axes, mode = 'nearest' , reshape=False)

        # for i in range(0, image.shape[2]):
        #     tmp_img_tr = cv2.resize(img [..., i], (width, height), interpolation=cv2.INTER_CUBIC)
        #     tmp_img_tr = np.expand_dims(tmp_img_tr, -1)
        #     img_tr = tmp_img_tr if img_tr is None else np.concatenate((img_tr, tmp_img_tr), -1)
        #
        #     m1_ = cv2.resize(msk [..., i], (width, height), interpolation=cv2.INTER_NEAREST)
        #     m1_ = np.expand_dims(m1_, -1)
        #     m1 = m1_ if m1 is None else np.concatenate((m1, m1_), -1)


        z[org_suffix] = img
        z[lab_suffix] = msk

        return z
    def data_augmentation1_org(self, z):
        org_suffix = 'image'
        image = z[org_suffix]
        max_percentage = 0.2
        dim1, dim2, dim3 = image.shape
        m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
        d1 = np.random.randint(-m1, m1)
        d2 = np.random.randint(-m2, m2)
        d3 = np.random.randint(-m3, m3)

        z[org_suffix] = self.transform_matrix_offset_center_3d_img(image, d1, d2, d3)

        return  z

    def transform_matrix_offset_center_3d_img(self, matrix, x, y, z):
        offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        return ndimage.interpolation.affine_transform(matrix, offset_matrix , mode = 'nearest')

    def transform_matrix_offset_center_3d_lb(self, matrix, x, y, z):
        offset_matrix = np.array([[1, 0, 0, x], [0, 1, 0, y], [0, 0, 1, z], [0, 0, 0, 1]])
        return ndimage.interpolation.affine_transform(matrix, offset_matrix, mode = 'nearest')

    def data_augmentation1_lab(self, z):
        org_suffix = 'image'
        lab_suffix = 'mask'

        imag = z[lab_suffix]
        mask = imag


        imag = z[org_suffix]
        image = imag



        max_percentage = 0.2
        dim1, dim2, dim3 = image.shape
        m1, m2, m3 = int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2), int(dim1 * max_percentage / 2)
        d1 = np.random.randint(-m1, m1)
        d2 = np.random.randint(-m2, m2)
        d3 = np.random.randint(-m3, m3)

        z[lab_suffix] = self.transform_matrix_offset_center_3d_lb(mask, d1, d2, d3)
        z[org_suffix] = self.transform_matrix_offset_center_3d_img(image, d1, d2, d3)

        return z


    def data_augmentation3(self, x, y, z):
        org_suffix = '_img.nii.gz'
        lab_suffix = '_lab.nii.gz'

        imag = x[org_suffix]
        image1 = imag[0, ..., 0]

        imag = y[org_suffix]
        image2 = imag[0, ..., 0]

        imag = z[org_suffix]
        image3 = imag[0, ..., 0]

        imag = z[lab_suffix]
        im1 = imag[0, ..., 0]
        im2 = imag[0, ..., 1]
        im3 = imag[0, ..., 2]
        im4 = imag[0, ..., 3]
        im5 = imag[0, ..., 4]

        seq = iaa.Sequential(
            [iaa.Affine(translate_px={"x": (-40, 40)})
                 ],
            random_order=True
        )
        seq.deterministic = True
        i1 = np.array([seq(image=i) for i in image1])
        i2 = np.array([seq(image=i) for i in image2])
        i3 = np.array([seq(image=i) for i in image3])

        i4 = np.array([seq(image=i) for i in im1])
        i5 = np.array([seq(image=i) for i in im2])
        i6 = np.array([seq(image=i) for i in im3])
        i7 = np.array([seq(image=i) for i in im4])
        i8 = np.array([seq(image=i) for i in im5])

        seq.deterministic = False

        x[org_suffix] = np.reshape(i1, [1, 128, 64, 64, 1])
        y[org_suffix] = np.reshape(i2, [1, 128, 64, 64, 1])

        z[org_suffix] = np.reshape(i3, [1, 128, 64, 64, 1])
        j1 = np.reshape(i4, [1, 128, 64, 64, 1])
        j2 = np.reshape(i5, [1, 128, 64, 64, 1])
        j3 = np.reshape(i6, [1, 128, 64, 64, 1])
        j4 = np.reshape(i7, [1, 128, 64, 64, 1])
        j5 = np.reshape(i8, [1, 128, 64, 64, 1])

        k1 = np.concatenate((j1, j2), -1)
        k2 = np.concatenate((k1, j3), -1)
        k3 = np.concatenate((k2, j4), -1)

        z[lab_suffix] = np.concatenate((k3, j5), -1)
        return x, y, z

    def restore(self, ckpt_path):
        nets = self.model.net if type(self.model.net) is list else [self.model.net]
        net_args = {}
        for i in range(len(nets)):
            net_args.update({'net%d' % i: nets[i]})
        ckpt = tf.train.Checkpoint(**net_args)
        ckpt.restore(ckpt_path)

    def eval(self, data_in_CT, data_in_MRI, data_in_CT_lab, **kwargs):
        batch_size = kwargs.get('batch_size', 1)
        print_str = kwargs.get('print_str', True)
        need_imgs = kwargs.get('need_imgs', False)
        if type(data_in_CT ) is dict:
            time_start = time.time()
            eval_dict = self.model.eval(data_in_CT, data_in_MRI ,data_in_CT_lab, **kwargs)
            time_cost = time.time() - time_start
            if print_str:
                print('Evaluation time cost is {:.1f}'.format(time_cost))
            return eval_dict

        # data_provider = U.dict_concat(data_in_CT, data_in_MRI)
        ndata = (data_in_CT.size + data_in_MRI.size)
        m = ndata // (batch_size *2)
        n = ndata % (batch_size *2)
        results = {}
        imgs1 = {}
        imgs2 = {}
        if print_str:
            print('Evaluate {} data:'.format(ndata))
        time_start = time.time()
        for i in range(m):

            #feed_dict_CT = data_in_CT(batch_size)
            #feed_dict_MRI = data_in_MRI(batch_size)
            #feed_dict_CT_lab = data_in_CT_lab(batch_size)

            #feed_dict_CT = self.data_augmentation1_org(feed_dict_CT)
            #feed_dict_MRI = self.data_augmentation1_org(feed_dict_MRI)
            #feed_dict_CT_lab = self.data_augmentation1_org_lab(feed_dict_CT_lab)

            sub_results = self.model.eval(data_in_CT(batch_size), data_in_MRI(batch_size),data_in_CT_lab(batch_size), **kwargs)
            sub_imgs1 = sub_results.pop('imgs1') if 'imgs1' in sub_results else None
            sub_imgs2 = sub_results.pop('imgs2') if 'imgs2' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs1 is not None:
                imgs1 = U.dict_append(imgs1, sub_imgs1)
                imgs2 = U.dict_append(imgs2, sub_imgs2)
            if print_str:
                print('evalated {} data'.format(batch_size * (i + 1)))
        if n > 0:
            sub_results = self.model.eval(data_in_CT(n), data_in_MRI(n),data_in_CT_lab(n),  **kwargs)
            sub_imgs1 = sub_results.pop('imgs1') if 'imgs1' in sub_results else None
            sub_imgs2 = sub_results.pop('imgs2') if 'imgs2' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs1 is not None:
                imgs1 = U.dict_append(imgs1, sub_imgs1)
                imgs2 = U.dict_append(imgs2, sub_imgs2)
            print('evalated {} data'.format(ndata))
        if print_str:
            time_cost = time.time() - time_start
            print('Time cost is {:.1f}'.format(time_cost))
            print('  {}'.format(U.dict_to_str(results)))
        if need_imgs:
            return results, imgs1, imgs2
        else:
            return results

    def eval_test(self, data_in_CT, **kwargs):
        batch_size = kwargs.get('batch_size', 30)
        print_str = kwargs.get('print_str', True)
        need_imgs = kwargs.get('need_imgs', False)
        if type(data_in_CT ) is dict:
            time_start = time.time()
            eval_dict = self.model.eval_test(data_in_CT, **kwargs)
            time_cost = time.time() - time_start
            #if print_str:
                #print('Evaluation time cost is {:.1f}'.format(time_cost))
            return eval_dict

        # data_provider = U.dict_concat(data_in_CT, data_in_MRI)
        ndata = (data_in_CT.size )
        m = ndata // (batch_size )
        n = ndata % (batch_size)
        results = {}
        imgs1 = {}
        imgs2 = {}
        if print_str:
            print('Evaluate {} data:'.format(ndata))
        time_start = time.time()
        for i in range(m):
            sub_results = self.model.eval_test(data_in_CT(batch_size), **kwargs)
            sub_imgs1 = sub_results.pop('imgs1') if 'imgs1' in sub_results else None
            sub_imgs2 = sub_results.pop('imgs2') if 'imgs2' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs1 is not None:
                imgs1 = U.dict_append(imgs1, sub_imgs1)
                imgs2 = U.dict_append(imgs2, sub_imgs2)
            if print_str:
                print('evalated {} data'.format(batch_size * (i + 1)))
        if n > 0:
            sub_results = self.model.eval_test(data_in_CT(n),  **kwargs)
            sub_imgs1 = sub_results.pop('imgs1') if 'imgs1' in sub_results else None
            sub_imgs2 = sub_results.pop('imgs2') if 'imgs2' in sub_results else None
            results = U.dict_concat(results, sub_results)
            if need_imgs and sub_imgs1 is not None:
                imgs1 = U.dict_append(imgs1, sub_imgs1)
                imgs2 = U.dict_append(imgs2, sub_imgs2)
            print('evalated {} data'.format(ndata))
        if print_str:
            time_cost = time.time() - time_start
            print('Time cost is {:.1f}'.format(time_cost))
            print('  {}'.format(U.dict_to_str(results)))
        if need_imgs:
            return results, imgs1, imgs2
        else:
            return results
    def predict(self, data_in, batch_size=1):
        if type(data_in) is dict:
            time_start = time.time()
            predictions = self.model.predict(data_in)
            time_cost = time.time() - time_start
            print('Prediction time cost is {:.1f}'.format(time_cost))
            return predictions
        data_provider = data_in
        ndata = data_provider.size
        m = ndata // (batch_size)
        n = ndata % (batch_size)
        predictions = None
        for _ in range(m):
            sub_predictions = self.model.predict(data_provider(batch_size))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        if n > 0:
            sub_predictions = self.model.predict(data_provider(n))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        return predictions


    def predict_test(self, data_in, batch_size=1):
        if type(data_in) is dict:
            time_start = time.time()
            predictions = self.model.predict(data_in)
            time_cost = time.time() - time_start
            print('Prediction time cost is {:.1f}'.format(time_cost))
            return predictions
        data_provider = data_in
        ndata = data_provider.size
        m = ndata // (batch_size)
        n = ndata % (batch_size)
        predictions = None
        for _ in range(m):
            sub_predictions = self.model.predict(data_provider(batch_size))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        if n > 0:
            sub_predictions = self.model.predict(data_provider(n))
            predictions = np.concatenate((predictions, sub_predictions),
                                         0) if predictions is not None else sub_predictions
        return predictions

    def _grads_add(self, grads, mini_grads):
        if grads is None:
            grads = mini_grads
        else:
            if type(grads) is not tuple:
                for i, g in enumerate(mini_grads):
                    if g is not None:
                        grads[i] += g
            else:
                for gi in range(len(grads)):
                    sub_grads = grads[gi]
                    sub_mini_grads = mini_grads[gi]
                    for i, g in enumerate(sub_mini_grads):
                        if g is not None:
                            sub_grads[i] += g
        return grads

    def _grads_div(self, grads, n):
        if n != 1:
            if type(grads) is not tuple:
                for i in range(len(grads)):
                    if grads[i] is not None:
                        grads[i] /= n
            else:
                for gi in range(len(grads)):
                    sub_grads = grads[gi]
                    for i in range(len(sub_grads)):
                        if sub_grads[i] is not None:
                            sub_grads[i] /= n
        return grads

