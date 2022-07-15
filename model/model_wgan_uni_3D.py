import numpy as np
import tensorflow as tf
from abc import ABCMeta, abstractmethod
import cv2
from utils import eval_methods as EM
from utils import loss_tf as LF
from utils import util_latent as U
from utils.process_methods import one_hot, min_max
from models.model import Model
from scipy.ndimage import zoom

class GANModel(Model):

    def __init__(self, net, x_suffix, y_suffix, m_suffix=None, g_alpha=10, g_lambda=10, g_beta=5, dropout=0, strategy=None):
        super().__init__(net)
        self._x_suffix = x_suffix
        self._y_suffix = y_suffix
        self._m_suffix = m_suffix

        self._beta = g_beta
        self._alpha = g_alpha
        self._lambda = g_lambda

        self.dropout = dropout
        self.strategy = strategy


    def get_grads(self, data_dict_CT, data_dict_MRI, feed_dict_CT_lab):
        data_dict = U.dict_concat(data_dict_CT, data_dict_MRI)
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._x_suffix]
        labs = feed_dict_CT_lab[self._y_suffix]
        #labs[labs == -1] = 0
        x_CT = feed_dict_CT_lab[self._x_suffix]
        x_MRI = data_dict_MRI[self._x_suffix]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            [_, o_seg]= self.net[0](x_CT, self.dropout, True)
            [_, mseg] = self.net[0](x_MRI, self.dropout, True)
            [gen_logits, _] = self.net[0](xs, self.dropout, True)

            out_seg = tf.nn.softmax(mseg, -1)

            # lab_t = tf.argmax(labs, -1)
            # seg_t = tf.argmax(out_seg, -1)
            #
            # lab_t_r = tf.dtypes.cast(tf.expand_dims(lab_t, axis=-1), dtype=tf.float32)
            # seg_t_r = tf.dtypes.cast(tf.expand_dims(seg_t, axis=-1), dtype=tf.float32)
            #
            # seg_norm =  tf.math.divide(seg_t_r, 4)
            # lab_norm = tf.math.divide(lab_t_r, 4)

            disc_gen_logits = self.net[1](out_seg[...,1:], self.dropout, True)
            disc_real_logits = self.net[1](labs[...,1:], self.dropout, True)

            gen_loss, _ ,_ ,_,_ = self._get_gen_loss(disc_gen_logits, gen_logits, ys
                                              , o_seg, labs)
            disc_loss = self._get_disc_loss(disc_real_logits, disc_gen_logits, gen_logits, ys,  True)

        # gen_loss = gen_loss

        gen_grads = gen_tape.gradient(gen_loss, self.net[0].trainable_variables)
        disc_grads = disc_tape.gradient(disc_loss, self.net[1].trainable_variables)

        # total_params = 0
        #
        # for v in self.net[1].trainable_variables:
        #     print(v)
        #     shape = v.get_shape()
        #     count = 1
        #     for dim in shape:
        #         count *= dim
        #         total_params += count
        # print("Total Params:%d" % total_params)
        return gen_grads, disc_grads

    def eval(self, data_dict_CT, data_dict_MRI, feed_dict_CT_lab, **kwargs):
        data_dict = U.dict_concat(data_dict_CT, data_dict_MRI)
        xs = data_dict[self._x_suffix]
        ys = data_dict[self._x_suffix]

        labs = feed_dict_CT_lab[self._y_suffix]
        x_CT = feed_dict_CT_lab[self._x_suffix]
        x_MRI = data_dict_MRI[self._x_suffix]
        #lab_idx = labs == -1
        #labs[labs == -1] = 0
        [_, o_seg] = self.net[0](x_CT, 0, False)
        [_, mseg] = self.net[0](x_MRI, 0, False)

        [gen_logits, _] = self.net[0](xs, 0., False)

        out_seg = tf.nn.softmax(mseg, -1)

        # lab_t = tf.argmax(labs, -1)
        # seg_t = tf.argmax(out_seg, -1)
        #
        # lab_t_r = tf.dtypes.cast(tf.expand_dims(lab_t, axis=-1), dtype=tf.float32)
        # seg_t_r = tf.dtypes.cast(tf.expand_dims(seg_t, axis=-1), dtype=tf.float32)
        #
        # seg_norm = tf.math.divide(seg_t_r, 4)
        # lab_norm = tf.math.divide(lab_t_r, 4)

        disc_gen_logits = self.net[1](out_seg[...,1:], self.dropout, False)
        disc_real_logits = self.net[1](labs[...,1:], self.dropout, False)

        # latent_loss = self._get_latent_loss(logits_enc_CT, logits_enc_MRI )
        total_gen_loss, gen_loss, seg_loss, mse_loss, latent_loss = self._get_gen_loss(disc_gen_logits, gen_logits, ys
                                                       , o_seg, labs)
        disc_loss = self._get_disc_loss(disc_real_logits, disc_gen_logits, gen_logits, ys,  False)
        #gen_mae = tf.reduce_mean(tf.losses.mae(gen_logits, ys))
        prob = tf.nn.softmax(o_seg, -1)
        pred = one_hot(np.argmax(prob, -1), list(range(labs.shape[-1])))

        #pred[lab_idx] = np.nan

        dice = EM.dice_coefficient(pred, labs)
        precision = EM.precision(pred, labs)
        recall = EM.recall(pred, labs)
        iou = EM.iou(pred, labs)

        eval_results = {'loss': total_gen_loss,
                        'gen_loss': gen_loss,
                        'disc_loss': disc_loss,
                        'gen_mse': mse_loss,
                        'seg_loss': seg_loss,
                        'latent_loss': latent_loss,
                        'dice': dice,
                        'precision': precision,
                        'recall': recall,
                        'iou': iou
                        }


        need_imgs = kwargs.get('need_imgs', None)
        if need_imgs is not None:
            eval_results.update({'imgs1': self._get_imgs_eval(x_CT, labs, prob)})
            eval_results.update({'imgs2': self._get_imgs_eval(xs, ys, gen_logits)})
        return eval_results


    def eval_test(self, data_dict, **kwargs):

        xs = data_dict[self._x_suffix]
        ys = data_dict[self._y_suffix]
        #lab_idx = ys == -1
        #ys[ys == -1] = 0
        [_,o_seg] = self.net[0](xs, 0, False)

        prob = tf.nn.softmax(o_seg, -1)
        pred = np.argmax(prob, -1)
        ys_ = np.argmax(ys, -1)
        # pred[lab_idx] = np.nan
        # dice_2d = EM.dice_coefficient(pred, ys)
        # s 1: 0 1 2 3 4
        # s 2: 0 1 2 3 (4)
        # s 3: 0 1 2 3 4
        width = 256
        height = 256
        dim = (width, height)
        resized_p = []
        resized_y = []
        for i in range(64):
            pred_i = pred[0, ..., i]
            resized_p.append(cv2.resize(pred_i, dim, interpolation=cv2.INTER_NEAREST))

            y_i = ys_[0, ..., i]
            resized_y.append(cv2.resize(y_i, dim, interpolation=cv2.INTER_NEAREST))

        pred_ = np.array(resized_p)
        ys_ = np.array(resized_y)


        pred = one_hot(ys_, list(range(ys.shape[-1])), masked=False)
        ys = one_hot(pred_, list(range(ys.shape[-1])), masked=False)

        # pred = np.resize(pred, (64, 256, 256, 5) )
        # ys = np.resize(ys, ( 64, 256, 256, 5))
        # # pred =
        # ys = np.transpose(ys, (1, 2, 0, 3))
        # pred = np.transpose(pred, (1, 2, 0, 3))
        #
        # pred = np.reshape (pred, [1, pred.shape[0], pred.shape[1], pred.shape[2], pred.shape[3]])
        # ys = np.reshape (ys, [1, ys.shape[0], ys.shape[1], ys.shape[2], ys.shape[3]])

        dice = EM.dice_coefficient(pred, ys)
        precision = EM.precision(pred, ys)
        recall = EM.recall(pred, ys)
        iou = EM.iou(pred, ys)
        # HDA = EM.hda(pred, ys)
        # assd = EM.assd_(pred, ys)
        p1 = np.argmax(pred, -1)
        l1 = np.argmax(ys, -1)

        # overlap_result, surface_distance_result = EM.Hausdorff_compute(p1[0, ...], l1[0, ...], (1.0, 1.0, 1.0))
        eval_results = {
                        'dice': dice,

                        'precision': precision,

                        'recall': recall,

                        'iou': iou,
                        # 'HD':HDA,
                        # 'assd':surface_distance_result[...,1]
                        #  'assd': assd
                        }
        return eval_results


    def predict(self, data_dict):
        [xout,latent,seg] = self.net[0](data_dict[self._x_suffix])
        return seg


    def _get_latent_loss(self, logits_CT, logits_MRI):

        loss_CT = tf.reduce_mean(logits_CT, 0)
        loss_MRI = tf.reduce_mean(logits_MRI, 0)

        sub_res = tf.abs(loss_CT - loss_MRI)

        return tf.reduce_mean(sub_res)


    def _get_gen_loss(self, disc_gen_logits, gen_logits, ys, o_seg, labs):
        #labs[labs == -1] = 0
        weight_map = np.zeros((tf.shape(ys)[0], tf.shape(ys)[1], tf.shape(ys)[2], tf.shape(ys)[3], tf.shape(ys)[4]), dtype=np.float)

        gen_loss = -tf.reduce_mean(disc_gen_logits)

        count_c1 = [len(v[v < 0]) for v in ys]
        count_c2 = [len(v[v > 0]) for v in ys]
        i = 0
        for io in range(0, tf.shape(ys)[0], 1):
            weight_map[i, :] = tf.where(ys[i, :] < 0, 1 - (count_c1[io] / (count_c1[io] + count_c2[io])), 1 - (count_c2[io] / (count_c1[io] + count_c2[io])))
            i = i + 1
        l1_loss = tf.reduce_mean(tf.compat.v1.losses.mean_squared_error(labels=ys, predictions=gen_logits, weights= weight_map))

        KL_loss =0

        # class_weights = tf.constant([[1.0, 10, 1.0, 1.0, 20]])
        # weights = tf.reduce_sum(class_weights * labs, axis=-1)
        # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=o_seg, labels=labs)
        # weighted_losses = unweighted_losses * weights
        # loss_map_seg = tf.reduce_mean(weighted_losses)

        # loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=o_seg, labels=labs)
        # weight_map = self.balance_weight_map(labs)
        # loss_map_seg = tf.reduce_mean(loss_map * weight_map)
        loss_map_seg = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=o_seg, labels=labs))
        return gen_loss +  self._alpha*l1_loss + 40*loss_map_seg  , gen_loss, loss_map_seg , l1_loss, KL_loss


    def _get_disc_loss(self, disc_real_logits, disc_gen_logits, gen_logits, ys, training):
        #ys[ys == -1] = 0
        temp_gen = gen_logits
        temp_ys = ys
        gen_logits = tf.concat([temp_gen, temp_gen],-1)
        gen_logits = tf.concat([gen_logits, temp_gen], -1)
        gen_logits = tf.concat([gen_logits, temp_gen],-1)
        # gen_logits = tf.concat([gen_logits, temp_gen], -1)
        # gen_logits = tf.concat([gen_logits, temp_gen],-1)
        # gen_logits = tf.concat([gen_logits, temp_gen],-1)
        # gen_logits = tf.concat([gen_logits, temp_gen],-1)
        # gen_logits = tf.concat([gen_logits, temp_gen],-1)
        #
        #
        ys = tf.concat([temp_ys, temp_ys], -1)
        ys = tf.concat([ys, temp_ys], -1)
        ys = tf.concat([ys, temp_ys], -1)
        # ys = tf.concat([ys, temp_ys], -1)
        # ys = tf.concat([ys, temp_ys], -1)
        # ys = tf.concat([ys, temp_ys], -1)
        # ys = tf.concat([ys, temp_ys], -1)
        # ys = tf.concat([ys, temp_ys], -1)


        alpha = tf.random.uniform(shape=[ys.shape[0], 1, 1, 1, 4], minval=0., maxval=1.)
        inter_sample = gen_logits* alpha + ys * (1 - alpha)

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(inter_sample)
            inter_score = self.net[1](inter_sample, self.dropout, training)
        gp_gradients = gp_tape.gradient(inter_score, inter_sample)
        gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3, 4]))
        gp_loss = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
        disc_loss = tf.reduce_mean(disc_gen_logits) - tf.reduce_mean(disc_real_logits)

        return disc_loss + gp_loss * self._lambda


    def gradient_penalty(self, gen_logits, ys):
        alpha = tf.random.uniform(shape=[ys.shape[0], 1, 1, 1, 1], minval=0., maxval=1.)
        differences = gen_logits - ys
        interpolates = ys + (alpha * differences)
        gradients = tf.gradients(self.net[1](interpolates), [interpolates])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2, 3, 4]))
        gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
        return gradient_penalty


    def _get_imgs_eval2(self, xs, ys, prob):
        img_dict = {}
        n_class = ys.shape[-1]
        for i in range(n_class):
            img = U.combine_3d_imgs_from_tensor([xs, prob[..., i]])
            img_dict.update({'class %d' % i: img})
        return img_dict


    def _get_imgs_eval(self, xs, ys, prob):
        img_dict = {}
        n_class = ys.shape[-1]
        for i in range(xs.shape[0]):
            _xs = np.transpose(xs[i], (2, 0, 1, 3))
            _ys = np.transpose(ys[i], (2, 0, 1, 3))
            _prob = np.transpose(prob[i], (2, 0, 1, 3))
            for j in range(n_class):
                img = U.combine_3d_imgs_from_tensor([_xs[...,0], _ys[...,j], _prob[..., j]])
                img_dict.update({'sub_%d_class_%d'%(i,j): img})

            argmax_ys = np.argmax(_ys, -1)
            argmax_prob = np.argmax(_prob, -1)
            img = U.combine_3d_imgs_from_tensor([_xs[...,0], argmax_ys, argmax_prob])
            img_dict.update({'sub_%d_argmax'%i: img})
        return img_dict


    def cross_entropy(self, logits, labels):
        loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
        return loss_map


    def balance_weight_map(self, labels, epsilon=1e-9):
        axis = tuple(range(np.ndim(labels) - 1)) if np.ndim(labels) > 1 else -1
        c = 1 / (np.sum(1 / (epsilon + np.sum(labels, axis=axis))))
        weight_map = np.sum(
            labels * np.tile(c / (epsilon + np.sum(labels, axis=axis, keepdims=True)), list(labels.shape[0:-1]) + [1]),
            axis=-1)
        return weight_map
