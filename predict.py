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
from net.fpn_net_lite import fpn_net_lite
from net.fpn_net import fpn_net
from dataset.bdd_daytime import bdd_daytime

from skimage import io
import cv2
import time

from others.msrcr import *
from others.clahe import *
from others.ying import Ying_2017_CAIP

tf.app.flags.DEFINE_string(
    'checkpoint_dir', 'D:\MyDownloads\\ssim1_psnr0',
    'The path to a checkpoint from which to fine-tune.')

# tf.app.flags.DEFINE_string(
#     'checkpoint_dir', 'H:\YangYiFanPackages\dark_aug_checkpoints\\fpn',
#     'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

input = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)
groundtruth = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)
global_step = tf.Variable(0, trainable=False, name='global_step')


noise_vis = False
real_night_compare = False


def build_graph(input):
    # output = u_net(input, is_training=False)
    output, self_attention_pairs, attentions= fpn_net_lite(input, is_training=False)
    # output, self_attention_pairs, attentions = fpn_net(input, is_training=False)

    return output

def main(_):
    output = build_graph(input)

    logger.info('Total trainable parameters:%s'%(str(np.sum([np.prod(v.get_shape().as_list()) \
                                                             for v in tf.trainable_variables()]))))

    saver = tf.train.Saver()

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        pd = bdd_daytime(batch_size=1, for_what='test', shuffle=False)
        # pd = bdd_daytime(batch_size=1, for_what='real_night_test', shuffle=True)

        if ckpt:
            logger.info('loading %s...'%str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('load %s success...'%str(ckpt.model_checkpoint_path))
        else:
            raise ValueError('checkpoint_dir may be wrong..')

        while True:
            raw_img, dark_img = pd.load_batch()
            time0 = time.time()
            out_img = sess.run(output, feed_dict={input:dark_img})
            print('time-'+str(round(time.time()-time0, 5)))
            raw_img = raw_img[0]
            dark_img = dark_img[0]
            out_img = out_img[0]

            raw_img = np.uint8(raw_img * 255)
            dark_img = np.uint8(dark_img * 255)
            out_img = np.uint8(np.clip(out_img, 0., 1.) * 255)

            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            dark_img = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)


            raw_img = cv2.resize(raw_img, (418, 278))
            dark_img = cv2.resize(dark_img, (418, 278))
            out_img = cv2.resize(out_img, (418,278))

            if noise_vis:
                msrcr_img = np.uint8(MSRCR(dark_img))
                msrcp_img = np.uint8(MSRCP(dark_img))
                ying_img = np.uint8(Ying_2017_CAIP(dark_img))

            elif real_night_compare:
                ying_img = np.uint8(Ying_2017_CAIP(raw_img))
                clahe_img = np.uint8(CLAHE(raw_img))

            # msrcr_img = cv2.resize(msrcr_img, (418, 278))
            # msrcp_img = cv2.resize(msrcp_img, (418, 278))
            # ying_img = cv2.resize(ying_img, (418, 278))

            cv2.imshow('groundtruth', raw_img)
            cv2.imshow('dark_img', dark_img)
            cv2.imshow('prediction', out_img)
            if noise_vis:
                cv2.imshow('msrcr_img', msrcr_img)
                cv2.imshow('msrcp_img', msrcp_img)
                cv2.imshow('ying_img', ying_img)
            elif real_night_compare:
                cv2.imshow('ying_img', ying_img)
                cv2.imshow('clahe_img', clahe_img)
            cv2.waitKey()
            cv2.destroyAllWindows()


if __name__ == '__main__':
    tf.app.run()

