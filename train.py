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

from net.u_net import u_net
from net.fpn_net import fpn_net
from net.fpn_net_lite import fpn_net_lite
from dataset.bdd_daytime import bdd_daytime


tf.app.flags.DEFINE_string(
    'checkpoint_dir', '',
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'train_dir', './checkpoint/fpn_lite',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_string(
    'summary_dir', './summary/fpn_lite',
    'Directory where checkpoints are written to.')

tf.app.flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

tf.app.flags.DEFINE_integer(
    'batch_size', 6, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'f_log_step', 20,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'f_summary_step', 20,
    'The frequency with which the model is saved, in step.')

tf.app.flags.DEFINE_integer(
    'f_save_step', 9999,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'f_eval_step', 20,
    'The frequency with which summaries are saved, in step.')

tf.app.flags.DEFINE_integer(
    'max_step', 50000,
    'The frequency with which summaries are saved, in step.')

FLAGS = tf.app.flags.FLAGS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

input = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)
groundtruth = tf.placeholder(shape=[None, 278, 418, 3], dtype=tf.float32)

global_step = tf.Variable(0, trainable=False, name='global_step')

lr = tf.placeholder(dtype=tf.float32)

def _smooth_l1(x):
    """Smoothed absolute function. Useful to compute an L1 smooth error.
    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)  ## smooth_l1
    return r

def build_graph(input):
    output, attention_pairs, attentions = fpn_net_lite(input, is_training=True)

    attention_decay = 5e-2
    attention_regularization = 0.
    for attention_pair in attention_pairs:
        attention_regularization += tf.reduce_mean(_smooth_l1(attention_pair[0] - attention_pair[1]))

    tf.summary.scalar('attention_regularization', attention_regularization)

    ## loss_1
    mse_loss = tf.reduce_sum(_smooth_l1(output - groundtruth)) / FLAGS.batch_size

    output = (output + 1.) * 255. / 2.
    gt_img = (groundtruth + 1.) * 255. / 2.

    psnr = tf.image.psnr(output, gt_img, max_val=255.)
    psnr_loss = tf.reduce_sum(1 / (psnr + 1e-8))

    ssmi_loss = tf.image.ssim_multiscale(output, gt_img, max_val=255.)
    ssmi_loss = tf.maximum(ssmi_loss, 1e-8)
    ssmi_loss = tf.reduce_sum(-tf.log(ssmi_loss))

    # LOSS_MSE #
    loss_1 = mse_loss

    # LOSS_SSMI_MSE #
    loss_2 = ssmi_loss*10000 + mse_loss

    # LOSS_SSMI_PSNR RECOMMEND#
    loss_3 = ssmi_loss + psnr_loss + attention_decay*attention_regularization##

    tf.summary.scalar('total_loss', loss_3)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        grads_and_vars = optimizer.compute_gradients(loss_3)
        ## clip the gradients ##
        capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var)
                      for grad, var in grads_and_vars]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    return output, loss_3, train_op

def main(_):
    output, loss, train_op = build_graph(input)

    logger.info('Total trainable parameters:%s'%(str(np.sum([np.prod(v.get_shape().as_list()) \
                                                             for v in tf.trainable_variables()]))))

    saver = tf.train.Saver(max_to_keep=5)

    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    init = tf.global_variables_initializer()
    merge_ops = tf.summary.merge_all()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        ## create a summary writer ##
        summary_dir = os.path.join(FLAGS.summary_dir)
        writer = tf.summary.FileWriter(FLAGS.summary_dir, sess.graph)

        pd = bdd_daytime(batch_size=FLAGS.batch_size, for_what='train', shuffle=True)

        if ckpt:
            logger.info('loading %s...'%str(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
            logger.info('load %s success...'%str(ckpt.model_checkpoint_path))
        else:
            sess.run(init)
            logger.info('Init Tf parameters success...')

        avg_loss = 0.
        current_step  = sess.run(global_step)
        while current_step < FLAGS.max_step:
            if current_step<FLAGS.max_step//3:
                learning_rate = FLAGS.learning_rate
            elif current_step<FLAGS.max_step*2//3:
                learning_rate = FLAGS.learning_rate / 10.
            else:
                learning_rate = FLAGS.learning_rate / 10.

            gt_imgs, train_imgs = pd.load_batch()

            update_op, m_ops, l, current_step = sess.run([train_op, merge_ops, loss, global_step],
                                                  feed_dict={input:train_imgs,
                                                             groundtruth:gt_imgs,
                                                             lr:learning_rate})

            if FLAGS.f_log_step != None:
                ## caculate average loss ##
                step = current_step % FLAGS.f_log_step
                avg_loss = (avg_loss * step + l) / (step + 1.)

                if current_step % FLAGS.f_log_step == FLAGS.f_log_step - 1:
                    ## print info ##
                    logger.info('Step%s loss:%s' % (str(current_step), str(avg_loss)))
                    avg_loss = 0.

            if FLAGS.f_summary_step != None:
                if current_step % FLAGS.f_summary_step == FLAGS.f_summary_step - 1:
                    ## summary ##
                    writer.add_summary(m_ops, current_step)


            if FLAGS.f_save_step != None:
                if current_step % FLAGS.f_save_step == FLAGS.f_save_step - 1:
                    ## save model ##
                    logger.info('Saving model...')
                    model_name = os.path.join(FLAGS.train_dir, 'dark_aug.model')
                    saver.save(sess, model_name, global_step=current_step)
                    logger.info('Save model sucess...')

        logger.info('Saving model...')
        model_name = os.path.join(FLAGS.train_dir, 'dark_aug_final.model')
        saver.save(sess, model_name, global_step=current_step)
        logger.info('Save model sucess...')


if __name__ == '__main__':
    tf.app.run()

