import time
import os
import numpy as np
import scipy.misc as scm

import vn

import tensorflow as tf
import argparse
import glob
import traceback

from denoisingdata import VnDenoisingData

import csv

def writeCsv(filename, rows, writetype='wb'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, writetype) as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, default='./configs/training.yaml')
    parser.add_argument('--data_config', type=str, default='./configs/data.yaml')

    args = parser.parse_args()
    checkpoint_config = tf.contrib.icg.utils.loadYaml(args.training_config, ['checkpoint_config'])
    data_config = tf.contrib.icg.utils.loadYaml(args.data_config, ['data_config'])

    epoch = 1000

    all_folders = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_folders = sorted([d for d in all_folders if os.path.isdir(d)])

    eval_file = checkpoint_config['log_dir'] + time.strftime('%Y-%m-%d--%H-%M-%S') + '_eval.csv'
    out_list = []

    save_output = True
    disp_slice_eval = False

    for suffix in all_folders:

        tf.reset_default_graph()
        suffix = suffix.split('/')[-1]
        print(suffix)
        # check the checkpoint directory
        ckpt_dir = checkpoint_config['log_dir'] + '/' + suffix + '/checkpoints/'
        eval_output_dir = checkpoint_config['log_dir'] + '/' + suffix + '/test/'

        with tf.Session() as sess:
            try:
                # load from checkpoint if required
                vn.utils.loadCheckpoint(sess, ckpt_dir, epoch=epoch)
            except Exception as e:
                print(traceback.print_exc())
                continue

            psnr_op = tf.get_collection('psnr_op')[0]
            ssim_op = tf.get_collection('ssim_op')[0]
            u_op = tf.get_collection('u_op')[0]
            u_var = tf.get_collection('u_var')
            f_var = tf.get_collection('f_var')
            g_var = tf.get_collection('g_var')

            # create data object
            data = VnDenoisingData(data_config, u_var=u_var, f_var=f_var, g_var=g_var)

             # Evaluate the performance
            print("Evaluating performance")
            eval_output_dir = checkpoint_config['log_dir'] + '/' + suffix + '/test/'
            if not os.path.exists(eval_output_dir):
                os.makedirs(eval_output_dir)

            psnr_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            ssim_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            time_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            for i in range(data.num_eval_images()):
                feed_dict = data.get_eval_feed_dict()
                eval_start_time = time.time()
                psnr_i, ssim_i, u_i = sess.run([psnr_op, ssim_op, u_op], feed_dict=feed_dict)
                time_eval[i] = time.time() - eval_start_time
                psnr_eval[i] = psnr_i
                ssim_eval[i] = ssim_i

                if disp_slice_eval:
                    print("{:4d}: {:.3f}dB {:.4f}".format(i, psnr_i, ssim_i))

                # save the output
                if save_output:
                    #print(u_i.shape)
                    scm.toimage(u_i[0, :, :, 0]*255, cmin=0, cmax=255).save(eval_output_dir + "result_{:d}.png".format(i))

            print("-----")
            print(" SUFFIX: {:s}".format(suffix))
            print(" AVG: {:.3f}  {:.4f}".format(np.mean(psnr_eval), np.mean(ssim_eval)))
            print("=====")
            print(" AVG inference time: {:.6f}s".format(np.mean(time_eval)))

            out_list.append([suffix, np.mean(psnr_eval), np.mean(ssim_eval)])
    out_list = [['suffix', 'psnr', 'ssim']]+sorted(out_list, key=lambda elem: (elem[0]))
    writeCsv(eval_file, out_list, 'w')
