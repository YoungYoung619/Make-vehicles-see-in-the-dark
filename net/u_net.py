"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
slim = tf.contrib.slim

def group_norm(x, G=32, esp=1e-5):
    # normalize
    # tranpose: [bs, h, w, c] to [bs, c, h, w] following the paper
    x = tf.transpose(x, [0, 3, 1, 2])
    N, C, H, W = x.get_shape().as_list()
    G = min(G, C)
    x = tf.reshape(x, [-1, G, C // G, H, W])
    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + esp)
    # per channel gamma and beta
    gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
    gamma = tf.reshape(gamma, [1, C, 1, 1])
    beta = tf.reshape(beta, [1, C, 1, 1])

    output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
    # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
    output = tf.transpose(output, [0, 2, 3, 1])
    return output

def lrelu(x):
    return tf.maximum(x * 0.2, x)

def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)

    return deconv_output

def u_net(input, is_training, norm_type='gn'):  # Unet
    assert norm_type in ['bn', 'gn']
    if norm_type == 'bn':
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params={'is_training': is_training}):
            conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
            conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
            pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

            conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
            conv2 = slim.conv2d(conv2, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv2_2')
            pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

            conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
            conv3 = slim.conv2d(conv3, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv3_2')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

            conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
            conv4 = slim.conv2d(conv4, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv4_2')
            pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

            conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
            conv5 = slim.conv2d(conv5, 512, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv5_2')

            up6 = upsample_and_concat(conv5, conv4, 256, 512)
            conv6 = slim.conv2d(up6, 256, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv6_1')
            conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

            up7 = upsample_and_concat(conv6, conv3, 128, 256)
            conv7 = slim.conv2d(up7, 128, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv7_1')
            conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')

            up8 = upsample_and_concat(conv7, conv2, 64, 128)
            conv8 = slim.conv2d(up8, 64, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv8_1')
            conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')

            up9 = upsample_and_concat(conv8, conv1, 32, 64)
            conv9 = slim.conv2d(up9, 32, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv9_1')
            conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

            conv10 = slim.conv2d(conv9, 16, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv10')
            conv11 = slim.conv2d(conv10, 16, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv11')

            conv12 = slim.conv2d(conv11, 3, [1, 1], rate=1, activation_fn=lrelu, scope='g_conv12')
            conv13 = slim.conv2d(conv12, 3, [3, 3], rate=1, activation_fn=tf.tanh, scope='g_conv13')
            return conv13
    elif norm_type == 'gn':
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=None, scope='g_conv1_1')
        conv1 = group_norm(conv1)
        conv1 = lrelu(conv1)
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        conv1 = group_norm(conv1)
        conv1 = lrelu(conv1)
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=None, scope='g_conv2_1')
        conv2 = group_norm(conv2)
        conv2 = lrelu(conv2)
        conv2 = slim.conv2d(conv2, 64, [1, 1], rate=1, activation_fn=None, scope='g_conv2_2')
        conv2 = group_norm(conv2)
        conv2 = lrelu(conv2)
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=None, scope='g_conv3_1')
        conv3 = group_norm(conv3)
        conv3 = lrelu(conv3)
        conv3 = slim.conv2d(conv3, 128, [1, 1], rate=1, activation_fn=None, scope='g_conv3_2')
        conv3 = group_norm(conv3)
        conv3 = lrelu(conv3)
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=None, scope='g_conv4_1')
        conv4 = group_norm(conv4)
        conv4 = lrelu(conv4)
        conv4 = slim.conv2d(conv4, 256, [1, 1], rate=1, activation_fn=None, scope='g_conv4_2')
        conv4 = group_norm(conv4)
        conv4 = lrelu(conv4)
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=None, scope='g_conv5_1')
        conv5 = group_norm(conv5)
        conv5 = lrelu(conv5)
        conv5 = slim.conv2d(conv5, 512, [1, 1], rate=1, activation_fn=None, scope='g_conv5_2')
        conv5 = group_norm(conv5)
        conv5 = lrelu(conv5)

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [1, 1], rate=1, activation_fn=None, scope='g_conv6_1')
        conv6 = group_norm(conv6)
        conv6 = lrelu(conv6)
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=None, scope='g_conv6_2')
        conv6 = group_norm(conv6)
        conv6 = lrelu(conv6)

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [1, 1], rate=1, activation_fn=None, scope='g_conv7_1')
        conv7 = group_norm(conv7)
        conv7 = lrelu(conv7)
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=None, scope='g_conv7_2')
        conv7 = group_norm(conv7)
        conv7 = lrelu(conv7)

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [1, 1], rate=1, activation_fn=None, scope='g_conv8_1')
        conv8 = group_norm(conv8)
        conv8 = lrelu(conv8)
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=None, scope='g_conv8_2')
        conv8 = group_norm(conv8)
        conv8 = lrelu(conv8)

        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [1, 1], rate=1, activation_fn=None, scope='g_conv9_1')
        conv9 = group_norm(conv9)
        conv9 = lrelu(conv9)
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=None, scope='g_conv9_2')
        conv9 = group_norm(conv9)
        conv9 = lrelu(conv9)

        conv10 = slim.conv2d(conv9, 16, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        conv10 = group_norm(conv10)
        conv10 = lrelu(conv10)
        conv11 = slim.conv2d(conv10, 16, [3, 3], rate=1, activation_fn=None, scope='g_conv11')
        conv11 = group_norm(conv11)
        conv11 = lrelu(conv11)

        conv12 = slim.conv2d(conv11, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv12')
        conv12 = group_norm(conv12)
        conv12 = lrelu(conv12)
        conv13 = slim.conv2d(conv12, 3, [3, 3], rate=1, activation_fn=None, scope='g_conv13')
        conv13 = group_norm(conv13)
        conv13 = tf.tanh(conv13)
        return conv13



if __name__ == '__main__':
    import numpy as np

    input = tf.placeholder(shape=[None, 139*2, 209*2, 3], dtype=tf.float32)
    out = u_net(input, is_training=True)
    print(str(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])))
    pass