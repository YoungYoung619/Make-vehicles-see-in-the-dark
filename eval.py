"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :
evaluate our method, the evaluation metrix include mse, nrmse, psnr, ssmi

Author：Team Li
"""
import tensorflow as tf
import numpy as np
import logging

from net.u_net import u_net
from dataset.bdd_daytime import bdd_daytime

from skimage import measure
import cv2


tf.app.flags.DEFINE_string(
    'checkpoint_dir', './checkpoint',
    'The path to a checkpoint from which to fine-tune.')

FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

input = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)
groundtruth = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)

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

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        pd = bdd_daytime(batch_size=1, for_what='test', shuffle=False)

        if ckpt:
            logger.info('loading %s...'%str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('load %s success...'%str(ckpt.model_checkpoint_path))
        else:
            raise ValueError('checkpoint_dir may be wrong..')

        t_ours_mse = 0.
        t_ours_nrmse = 0.
        t_ours_psnr = 0.
        t_ours_ssmi = 0.

        for i in range(pd.images_number()):
            raw_img, dark_img = pd.load_batch()
            out_img = sess.run(output, feed_dict={input:dark_img})
            raw_img = raw_img[0]
            # dark_img = dark_img[0]
            out_img = out_img[0]

            raw_img = np.uint8((raw_img + 1.) * 255 / 2)
            # dark_img = np.uint8((dark_img + 1.) * 255 / 2)
            out_img = np.uint8((out_img + 1.) * 255 / 2)
            raw_img = cv2.resize(raw_img, (418, 278))

            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_RGB2BGR)
            # dark_img = cv2.cvtColor(dark_img, cv2.COLOR_RGB2BGR)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)


            ours_mse = measure.compare_mse(raw_img, out_img)
            ours_nrmse = measure.compare_nrmse(raw_img, out_img)
            ours_psnr = measure.compare_psnr(raw_img, out_img)
            ours_ssmi = measure.compare_ssim(raw_img, out_img, multichannel=True)
            # logger.info('Ours-- mse:%f nrmse:%f psnr:%f ssmi:%f' % (ours_mse, ours_nrmse, ours_psnr, ours_ssmi))

            t_ours_mse += ours_mse
            t_ours_nrmse += ours_nrmse
            t_ours_psnr += ours_psnr
            t_ours_ssmi += ours_ssmi

            if i%50 == 0:
                logger.info('eval: [%d:%d]'%(i, pd.images_number()))

        t_n = pd.images_number()
        logger.info('Ours-- mse:%.2f nrmse:%.2f psnr:%.2f ssmi:%.2f' % (t_ours_mse/t_n, t_ours_nrmse/t_n,
                                                                        t_ours_psnr/t_n, t_ours_ssmi/t_n))


if __name__ == '__main__':
    tf.app.run()
