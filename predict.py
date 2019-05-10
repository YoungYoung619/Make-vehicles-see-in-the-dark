"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import tensorflow as tf
import numpy as np
import logging
import os
import glob

from net.u_net import u_net
from dataset.bdd_daytime import bdd_daytime

from skimage import io, exposure, transform
import cv2

tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoint/2',
    'The path to a checkpoint from which to fine-tune.')


FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

input = tf.placeholder(shape=[None, 139*2, 209*2, 3], dtype=tf.float32)
groundtruth = tf.placeholder(shape=[None, 139*2, 209*2, 3], dtype=tf.float32)
# input = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
# groundtruth = tf.placeholder(shape=[None, 256, 256, 3], dtype=tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')


def build_graph(input):
    output = u_net(input, is_training=False)

    return output

def main(_):
    output = build_graph(input)

    logger.info('Total trainable parameters:%s'%(str(np.sum([np.prod(v.get_shape().as_list()) \
                                                             for v in tf.trainable_variables()]))))

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        pd = bdd_daytime(batch_size=1, for_what='test', shuffle=True)

        if ckpt:
            logger.info('loading %s...'%str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('load %s success...'%str(ckpt.model_checkpoint_path))
        else:
            raise ValueError('checkpoint_dir may be wrong..')

        while True:
            raw_img, dark_img = pd.load_batch()
            out_img = sess.run(output, feed_dict={input:dark_img})
            raw_img = raw_img[0]
            dark_img = dark_img[0]
            out_img = out_img[0]

            raw_img = np.uint8((raw_img + 1.) * 255 / 2)
            dark_img = np.uint8((dark_img + 1.) * 255 / 2)
            out_img = np.uint8((out_img + 1.) * 255 / 2)

            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            dark_img = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

            raw_img = cv2.resize(raw_img, (418, 278))
            dark_img = cv2.resize(dark_img, (418, 278))
            out_img = cv2.resize(out_img, (418,278))

            ##other method
            gamma_img = exposure.adjust_gamma(raw_img, 0.5)

            cv2.imshow('groundtruth', raw_img)
            cv2.imshow('dark_img', dark_img)
            cv2.imshow('prediction', out_img)

            cv2.imshow('gamma', gamma_img.astype(np.uint8))
            cv2.waitKey()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tf.app.run()

