import tensorflow as tf
import collections

weight_decay=1e-4
keep_prob = 1.

def relu(x, name='relu6'):
    return tf.nn.relu6(x, name)

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


def conv2d(input_, output_dim, k_h, k_w, d_h, d_w, stddev=0.02, name='conv2d', bias=False):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
              regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
              initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')
        if bias:
            biases = tf.get_variable('bias', [output_dim], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        return conv


def conv2d_block(input, out_dim, k, s, is_train, name):
    with tf.name_scope(name), tf.variable_scope(name):
        net = conv2d(input, out_dim, k, k, s, s, name='conv2d')
        net = group_norm(net)
        net = relu(net)
        return net


def conv_1x1(input, output_dim, name, bias=False):
    with tf.name_scope(name):
        conv = conv2d(input, output_dim, 1,1,1,1, stddev=0.02, name=name, bias=bias)
        conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        return conv


def pwise_block(input, output_dim, is_train, name, bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        out=conv_1x1(input, output_dim, bias=bias, name='pwb')
        out=group_norm(out)
        out=relu(out)
        return out


def dwise_conv(input, k_h=3, k_w=3, channel_multiplier= 1, strides=[1,1,1,1],
               padding='SAME', stddev=0.02, name='dwise_conv', bias=False):
    with tf.variable_scope(name):
        in_channel=input.get_shape().as_list()[-1]
        w = tf.get_variable('w', [k_h, k_w, in_channel, channel_multiplier],
                        regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.depthwise_conv2d(input, w, strides, padding, rate=None,name=None,data_format=None)
        if bias:
            biases = tf.get_variable('bias', [in_channel*channel_multiplier], initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        return conv


def res_block(input, expansion_ratio, output_dim, stride, is_train, name, bias=False, shortcut=True):
    with tf.name_scope(name), tf.variable_scope(name):
        # pw
        bottleneck_dim=round(expansion_ratio*input.get_shape().as_list()[-1])
        net = conv_1x1(input, bottleneck_dim, name='pw', bias=bias)
        net = group_norm(net)
        net = relu(net)
        # dw
        net = dwise_conv(net, strides=[1, stride, stride, 1], name='dw', bias=bias)
        net = group_norm(net)
        net = relu(net)
        # pw & linear
        net = conv_1x1(net, output_dim, name='pw_linear', bias=bias)
        net = group_norm(net)

        # element wise add, only for stride==1
        if shortcut and stride == 1:
            in_dim=int(input.get_shape().as_list()[-1])
            if in_dim != output_dim:
                ins=conv_1x1(input, output_dim, name='ex_dim')
                net=ins+net
            else:
                net=input+net

        return net


def separable_conv(input, k_size, output_dim, stride, pad='SAME', channel_multiplier=1, name='sep_conv', bias=False):
    with tf.name_scope(name), tf.variable_scope(name):
        in_channel = input.get_shape().as_list()[-1]
        dwise_filter = tf.get_variable('dw', [k_size, k_size, in_channel, channel_multiplier],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))

        pwise_filter = tf.get_variable('pw', [1, 1, in_channel*channel_multiplier, output_dim],
                  regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                  initializer=tf.truncated_normal_initializer(stddev=0.02))
        strides = [1,stride, stride,1]

        conv=tf.nn.separable_conv2d(input,dwise_filter,pwise_filter,strides,padding=pad, name=name)
        if bias:
            biases = tf.get_variable('bias', [output_dim],initializer=tf.constant_initializer(0.0))
            conv = tf.nn.bias_add(conv, biases)

        conv = tf.nn.dropout(conv, keep_prob=keep_prob)
        return conv


def global_avg(x):
    with tf.name_scope('global_avg'):
        net=tf.layers.average_pooling2d(x, x.get_shape()[1:-1], 1)
        return net


def flatten(x):
    #flattened=tf.reshape(input,[x.get_shape().as_list()[0], -1])  # or, tf.layers.flatten(x)
    return tf.contrib.layers.flatten(x)


def pad2d(inputs, pad=(0, 0), mode='CONSTANT'):
    paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
    net = tf.pad(inputs, paddings, mode=mode)
    return net

def backbone_lite(inputs, is_training):
    """mobilenetv2( deleted the global average pooling )
    Args:
        inputs: a tensor with the shape (bs, h, w, c)
        is_training: indicate whether to train or test
    Return:
        all the end point.
    """
    endPoints = collections.OrderedDict()
    exp = 6  # expansion ratio
    with tf.variable_scope('mobilenetv2'):
        net = conv2d_block(inputs, 32, 3, 2, is_training, name='conv1_1')  # size/2
        endPoints['conv_1'] = net

        net = res_block(net, 1, 16, 1, is_training, name='res2_1')
        net = res_block(net, exp, 24, 2, is_training, name='res3_1')  # size/4
        net = res_block(net, exp, 24, 1, is_training, name='res3_2')
        endPoints['conv_2'] = net

        net = res_block(net, exp, 32, 2, is_training, name='res4_1')  # size/8
        net = res_block(net, exp, 32, 1, is_training, name='res4_2')
        net = res_block(net, exp, 32, 1, is_training, name='res4_3')
        endPoints['conv_3'] = net

        net = res_block(net, exp, 64, 2, is_training, name='res5_1')  # size/16
        net = res_block(net, exp, 64, 1, is_training, name='res5_2')
        net = res_block(net, exp, 64, 1, is_training, name='res5_3')
        net = res_block(net, exp, 64, 1, is_training, name='res5_4')
        endPoints['conv_4'] = net

        net = res_block(net, exp, 96, 1, is_training, name='res6_1')
        net = res_block(net, exp, 96, 1, is_training, name='res6_2')
        net = res_block(net, exp, 96, 1, is_training, name='res6_3')
        endPoints['conv_5'] = net

    return endPoints