import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers, initializers


class Generator(Model):
    def __init__(self, n_class, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True,
                 padding='SAME', concat_or_add='concat'):
        super().__init__()
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()
        self.dw1_layers = dict()

        for layer in range(n_layer):
            filters = 2 ** layer * root_filters
            dict_key = str(n_layer - layer - 1)
            if layer == 0:
               Dw1 = _DownSev(filters, 7, 'dw1_%d' % layer)
               self.dw1_layers[dict_key] = Dw1

            dw = _DownSampling(filters, kernal_size, 'dw_%d' % layer, use_bn, use_res)
            self.dw_layers[dict_key] = dw
            if layer < n_layer - 1:
                pool = layers.MaxPool2D(pool_size, padding=padding)
                self.max_pools[dict_key] = pool

        for layer in range(n_layer - 2, -1, -1):
            filters = 2 ** (layer + 1) * root_filters
            dict_key = str(n_layer - layer - 1)
            up = _UpSampling(filters, kernal_size, pool_size, 'up_%d' % layer, concat_or_add)
            self.up_layers[dict_key] = up

        stddev = np.sqrt(2 / (kernal_size ** 2 * root_filters))
        # stddev = 0.02
        self.conv_out1 = layers.Conv2D(1, 1, padding=padding, use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      name='conv_out1')
        self.conv_out2 = layers.Conv2D(5, 1, padding=padding, use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      name='conv_out2')

        # self.conv_out3 = layers.Conv2D(2048, 1, padding=padding, use_bias=False,
        #                               kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
        #                               name='conv_out3')


    @tf.function
    def __call__(self, x_in, dw_tensors=[],z=[] , drop_rate=0, training=False):
        n_layer = len(self.dw_layers)

        # if (tf.shape(z)[0] == 0):
        dw_tensors = dict()
        dw1_tensors = dict()
        x = x_in
        dict_key = str(3)
        dw1_tensors[dict_key] = self.dw1_layers[dict_key](x, drop_rate, training)
        x = dw1_tensors[dict_key]

        for i in range(n_layer):
            dict_key = str(n_layer - i - 1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, drop_rate, training)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)
        # else:
        #     x = z
        # xout = self.conv_out3(x)
        # mean = xout[..., :1024]
        # stdv = 1e-6 + tf.nn.softplus(xout[..., 1024:])
        latent = x
        for i in range(n_layer - 2, -1, -1):
            dict_key = str(n_layer - i - 1)
            x = self.up_layers[dict_key](x, dw_tensors[dict_key], drop_rate, training)

        x_o1 = x
        x_o2 = x

        x_o1 = self.conv_out1(x_o1)
        x1 = tf.nn.leaky_relu(x_o1)

        x_o2 = self.conv_out2(x_o2)
        x2 = tf.nn.leaky_relu(x_o2)

        return x1, latent, x2, dw_tensors

class _DownSev(Model):
    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=True):
        super(_DownSev, self).__init__(name=name)
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        self.conv1 = layers.Conv2D(filters, 7, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')

    def call(self, input_tensor, keep_prob, train_phase):
        # conv1
        x = self.conv1(input_tensor)
        x = tf.nn.dropout(x, keep_prob)
        x = tf.nn.leaky_relu(x)
        return x

class Gen_encoder(Model):
    def __init__(self, n_class, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True,
                 padding='SAME', concat_or_add='concat'):
        super().__init__()
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()
        self.dw1_layers = dict()

        for layer in range(n_layer):
            filters = 2 ** layer * root_filters
            dict_key = str(n_layer - layer - 1)
            if layer == 0:
               Dw1 = _DownSev(filters, 7, 'dw1_%d' % layer)
               self.dw1_layers[dict_key] = Dw1

            dw = _DownSampling(filters, kernal_size, 'dw_%d' % layer, use_bn, use_res)
            self.dw_layers[dict_key] = dw
            if layer < n_layer - 1:
                pool = layers.MaxPool2D(pool_size, padding=padding)
                self.max_pools[dict_key] = pool


    @tf.function
    def __call__(self, x_in, drop_rate=0, training=False):
        dw_tensors = dict()
        dw1_tensors = dict()
        x = x_in
        dict_key = str(3)
        dw1_tensors[dict_key] = self.dw1_layers[dict_key](x, drop_rate, training)
        x = dw1_tensors[dict_key]
        n_layer = len(self.dw_layers)
        for i in range(n_layer):
            dict_key = str(n_layer - i - 1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, drop_rate, training)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)


        return x, dw_tensors

class Gen_decoder(Model):
    def __init__(self, n_class, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True,
                 padding='SAME', concat_or_add='concat'):
        super().__init__()
        self.up_layers = dict()


        for layer in range(n_layer - 2, -1, -1):
            filters = 2 ** (layer + 1) * root_filters
            dict_key = str(n_layer - layer - 1)
            up = _UpSampling(filters, kernal_size, pool_size, 'up_%d' % layer, concat_or_add)
            self.up_layers[dict_key] = up

        stddev = np.sqrt(2 / (kernal_size ** 2 * root_filters))
        # stddev = 0.02
        self.conv_out = layers.Conv2D(1, 1, padding=padding, use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      name='conv_out')


    @tf.function
    def __call__(self, x_in, dw_tensors, drop_rate=0, training=False):
        #dw_tensors = dict()

        x = x_in
        n_layer = len(dw_tensors)

        for i in range(n_layer - 2, -1, -1):
            dict_key = str(n_layer - i - 1)
            x = self.up_layers[dict_key](x, dw_tensors[dict_key], drop_rate, training)

        x = self.conv_out(x)
        xg = tf.nn.leaky_relu(x)


        return xg

class Discriminator(Model):
    def __init__(self, n_class, n_layer, root_filters, kernal_size=3, pool_size=2, use_bn=True, use_res=True,
                 padding='SAME', concat_or_add='concat'):
        super().__init__()
        self.dw_layers = dict()
        self.up_layers = dict()
        self.max_pools = dict()
        self.dw1_layers = dict()

        for layer in range(n_layer):
            filters = 2 ** layer * root_filters
            dict_key = str(n_layer - layer - 1)
            if layer == 0:
               Dw1 = _DownSev(filters, 7, 'dw1_%d' % layer)
               self.dw1_layers[dict_key] = Dw1

            dw = _DownSampling_disc(filters, kernal_size, 'dw_%d' % layer, use_bn, use_res)
            self.dw_layers[dict_key] = dw
            if layer < n_layer - 1:
                pool = layers.MaxPool2D(pool_size, padding=padding)
                self.max_pools[dict_key] = pool
        stddev = np.sqrt(2 / (kernal_size ** 2 * root_filters))
        # stddev = 0.02
        self.conv_out = layers.Conv2D(1, 1, padding=padding, use_bias=False,
                                      kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                      name='conv_out')

    @tf.function
    def __call__(self, x_in, drop_rate=0, training=False):
        dw_tensors = dict()
        dw1_tensors = dict()
        x = x_in
        dict_key = str(3)
        dw1_tensors[dict_key] = self.dw1_layers[dict_key](x, drop_rate, training)
        x = dw1_tensors[dict_key]
        n_layer = len(self.dw_layers)
        for i in range(n_layer):
            dict_key = str(n_layer - i - 1)
            dw_tensors[dict_key] = self.dw_layers[dict_key](x, drop_rate, training)
            x = dw_tensors[dict_key]
            if i < len(self.max_pools):
                x = self.max_pools[dict_key](x)


        x = self.conv_out(x)
        xg = tf.nn.leaky_relu(x)


        return xg

class _DownSampling_disc(Model):
    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=False):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        # stddev = 0.02
        self.conv1 = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')

    def __call__(self, x_in, drop_rate, training=True):
        # conv1

        x = x_in
        x = self.conv1(x)
        x = tf.nn.dropout(x, drop_rate)
        x = tf.nn.leaky_relu(x)

        return x

class _DownSampling(Model):
    def __init__(self, filters, kernel_size, name, use_bn=True, use_res=True, padding='SAME', use_bias=False):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        # stddev = 0.02
        self.conv1 = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')
        self.conv2 = layers.Conv2D(filters, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')

        if use_bn:
            self.bn1 = layers.BatchNormalization(momentum=0.99, name='bn1')
            self.bn2 = layers.BatchNormalization(momentum=0.99, name='bn2')
        if use_res:
            self.res = _Residual('res')

    def __call__(self, x_in, drop_rate, training=True):
        # conv1

        x = x_in
        x = self.conv1(x)
        x1 = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x1 = self.bn1(x1, training=training)
        x = tf.nn.leaky_relu(x1)

        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn2(x, training=training)

        # if x_in.shape[-1] < x.shape[-1]:
        #     x_in = tf.concat([x_in, tf.zeros(list(x_in.shape[:-1]) + [x.shape[-1] - x_in.shape[-1]])], axis=-1)

        x = x + x1 + x_in
        if self.use_res:
           x = self.res(x_in, x)
        x = tf.nn.leaky_relu(x)

        x = tf.concat((x, x_in), -1)

        return x



class _UpSampling(Model):
    def __init__(self, filters, kernel_size, pool_size, name, concat_or_add='concat', use_bn=True, use_res=True,
                 padding='SAME', use_bias=False):
        super().__init__(name=name)
        self.use_bn = use_bn
        self.use_res = use_res
        self.concat_or_add = concat_or_add
        stddev = np.sqrt(2 / (kernel_size ** 2 * filters))
        self.deconv = layers.Conv2DTranspose(filters // 2, kernel_size, strides=pool_size, padding=padding,
                                             use_bias=use_bias,
                                             kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                             name='deconv')
        self.conv1 = layers.Conv2D(filters // 2, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv1')
        self.conv2 = layers.Conv2D(filters // 2, kernel_size, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv2')
        self.conv3 = layers.Conv2D(filters // 2, 1, padding=padding, use_bias=use_bias,
                                   kernel_initializer=initializers.TruncatedNormal(stddev=stddev),
                                   name='conv3')

        if use_bn:
            self.bn_deconv = layers.BatchNormalization(momentum=0.99, name='bn_deconv')
            self.bn1 = layers.BatchNormalization(momentum=0.99, name='bn1')
            self.bn2 = layers.BatchNormalization(momentum=0.99, name='bn2')
            self.bn3 = layers.BatchNormalization(momentum=0.99, name='bn3')
        if use_res:
            self.res = _Residual('res')

    def __call__(self, x_in, x_dw, drop_rate, training):
        # deconv
        x = self.deconv(x_in)
        if self.use_bn:
            x = self.bn_deconv(x, training=training)
        x = tf.nn.leaky_relu(x)

        # skip connection
        if self.concat_or_add == 'concat':
            x = tf.concat((x_dw, x), -1)
        elif self.concat_or_add == 'add':
            x = x_dw + x
        else:
            raise Exception('Wrong concatenate method!')

        res_in = x
        # conv1
        x = self.conv1(x)
        x1 = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x1 = self.bn1(x1, training=training)
        x = tf.nn.leaky_relu(x1)

        # conv2
        x = self.conv2(x)
        x = tf.nn.dropout(x, drop_rate)
        if self.use_bn:
            x = self.bn2(x, training=training)
        if self.use_res:
           x = self.res(res_in, x)
        x3 = self.conv3(res_in)
        if self.use_bn:
            x3 = self.bn3(x3, training=training)

        x = x + x1 + x3
        x = tf.nn.leaky_relu(x)

        return x


class _Residual(Model):
    def __init__(self, name_scope):
        super(_Residual, self).__init__(name=name_scope)

    def call(self, x1, x2):
        if x1.shape[-1] < x2.shape[-1]:
            x = tf.concat([x1, tf.zeros(list(x1.shape[:-1]) + [x2.shape[-1] - x1.shape[-1]])], axis=-1)
        else:
            x = x1[..., :x2.shape[-1]]
        x = x + x2
        return x

