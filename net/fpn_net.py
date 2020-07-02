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

def fpn_net(input, is_training, norm_type='gn'):  # Unet
    with slim.arg_scope([slim.conv2d], padding='SAME',
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(2e-5)):
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

        attention_2 = tf.reduce_sum(conv2, axis=-1, keep_dims=True)

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=None, scope='g_conv3_1')
        conv3 = group_norm(conv3)
        conv3 = lrelu(conv3)
        conv3 = slim.conv2d(conv3, 128, [1, 1], rate=1, activation_fn=None, scope='g_conv3_2')
        conv3 = group_norm(conv3)
        conv3 = lrelu(conv3)
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        attention_3 = tf.reduce_sum(conv3, axis=-1, keep_dims=True)
        imitation_3_for_2 = tf.image.resize(attention_3, size=tuple(attention_2.get_shape().as_list()[1:3]))

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=None, scope='g_conv4_1')
        conv4 = group_norm(conv4)
        conv4 = lrelu(conv4)
        conv4 = slim.conv2d(conv4, 256, [1, 1], rate=1, activation_fn=None, scope='g_conv4_2')
        conv4 = group_norm(conv4)
        conv4 = lrelu(conv4)
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        attention_4 = tf.reduce_sum(conv4, axis=-1, keep_dims=True)
        imitation_4_for_3 = tf.image.resize(attention_4, size=tuple(attention_3.get_shape().as_list()[1:3]))

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=None, scope='g_conv5_1')
        conv5 = group_norm(conv5)
        conv5 = lrelu(conv5)
        conv5 = slim.conv2d(conv5, 512, [1, 1], rate=1, activation_fn=None, scope='g_conv5_2')
        conv5 = group_norm(conv5)
        conv5 = lrelu(conv5)

        attention_5 = tf.reduce_sum(conv5, axis=-1, keep_dims=True)
        imitation_5_for_4 = tf.image.resize(attention_5, size=tuple(attention_4.get_shape().as_list()[1:3]))

        ##  FPN
        fpn_channel = 256
        up5 = slim.conv2d(conv5, fpn_channel, [1, 1], rate=1, activation_fn=None, scope='fpn_conv5')

        ## expand the height, width, and combine with the conv4
        f_size = conv4.get_shape().as_list()[1:3]
        fpn_feat6 = tf.image.resize(up5, f_size)
        fpn_conv4 = slim.conv2d(conv4, fpn_channel, [1, 1], rate=1, activation_fn=None, scope='fpn_conv4')
        up6 = fpn_feat6 + fpn_conv4
        up6 = group_norm(up6)
        up6 = lrelu(up6)

        f_size = conv3.get_shape().as_list()[1:3]
        fpn_feat7 = tf.image.resize(up6, f_size)
        fpn_conv3 = slim.conv2d(conv3, fpn_channel, [1, 1], rate=1, activation_fn=None, scope='fpn_conv3')
        up7 = fpn_feat7 + fpn_conv3
        up7 = group_norm(up7)
        up7 = lrelu(up7)

        f_size = conv2.get_shape().as_list()[1:3]
        fpn_feat8 = tf.image.resize(up7, f_size)
        fpn_conv2 = slim.conv2d(conv2, fpn_channel, [1, 1], rate=1, activation_fn=None, scope='fpn_conv2')
        up8 = fpn_feat8 + fpn_conv2
        up8 = group_norm(up8)
        up8 = lrelu(up8)

        f_size = conv1.get_shape().as_list()[1:3]
        fpn_feat9 = tf.image.resize(up8, f_size)
        fpn_conv1 = slim.conv2d(conv1, fpn_channel, [1, 1], rate=1, activation_fn=None, scope='fpn_conv1')
        up9 = fpn_feat9 + fpn_conv1
        up9 = group_norm(up9)
        up9 = lrelu(up9)

        conv9 = slim.conv2d(up9, 128, [1, 1], rate=1, activation_fn=None, scope='g_conv9_1')
        conv9 = group_norm(conv9)
        conv9 = lrelu(conv9)
        conv9 = slim.conv2d(conv9, 128, [3, 3], rate=1, activation_fn=None, scope='g_conv9_2')
        conv9 = group_norm(conv9)
        conv9 = lrelu(conv9)

        conv10 = slim.conv2d(conv9, 32, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        conv10 = group_norm(conv10)
        conv10 = lrelu(conv10)
        conv11 = slim.conv2d(conv10, 32, [3, 3], rate=1, activation_fn=None, scope='g_conv11')
        conv11 = group_norm(conv11)
        conv11 = lrelu(conv11)

        conv12 = slim.conv2d(conv11, 3, [3, 3], rate=1, activation_fn=None, scope='g_conv12')
        conv12 = group_norm(conv12)
        conv12 = lrelu(conv12)
        conv13 = slim.conv2d(conv12, 3, [1, 1], rate=1, activation_fn=None, scope='g_conv13')
        conv13 = group_norm(conv13)
        conv13 = tf.sigmoid(conv13)

        attentions = [attention_2, attention_3, attention_4, attention_5]
        self_attention_pairs = [(attention_2, imitation_3_for_2), (attention_3, imitation_4_for_3), (attention_4, imitation_5_for_4)]
        return conv13, self_attention_pairs, attentions


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

if __name__ == '__main__':
    import numpy as np

    with tf.Graph().as_default() as graph:
        input = tf.placeholder(shape=[None, 139*2, 209*2, 3], dtype=tf.float32)
        out = fpn_net(input, is_training=True)

        stats_graph(graph)
    pass