import os
import sys
import argparse
import glob
import traceback
import time

import vn
import tensorflow as tf
import numpy as np
from scipy.misc import imread, toimage

parser = argparse.ArgumentParser(description='denoise a given input image using a model')
parser.add_argument('image_name', type=str, help='image file to denoise')
parser.add_argument('model_name', type=str, help='name of the model in the log dir')
parser.add_argument('-o', dest='output_name', type=str, default='result.jpg', help='output name')
parser.add_argument('-s', dest='sigma', type=int, default=0, help='additional gaussian noise (default 0)')
parser.add_argument('--training_config', type=str, default='./configs/training.yaml', help='training config file')

if __name__ == '__main__':
    # parse the input arguments
    args = parser.parse_args()
    # image and model
    image_name = args.image_name
    model_name = args.model_name

    sigma = args.sigma

    output_name = args.output_name

    # load the image
    image = imread(image_name, mode='L')[np.newaxis, :, :, np.newaxis].astype(np.float32) / 255
    noisy = image + sigma/255. * np.random.randn(*image.shape)

    # load the model
    checkpoint_config = tf.contrib.icg.utils.loadYaml(args.training_config, ['checkpoint_config'])

    all_models = glob.glob(checkpoint_config['log_dir'] + '/*')
    all_models = sorted([d.split('/')[-1] for d in all_models if os.path.isdir(d)])

    if not model_name in all_models:
        print('model not found in "{}"'.format(checkpoint_config['log_dir']))
        sys.exit(-1)

    ckpt_dir = checkpoint_config['log_dir'] + '/' + model_name + '/checkpoints/'
    eval_output_dir = checkpoint_config['log_dir'] + '/' + model_name + '/test/'

    with tf.Session() as sess:
        try:
            # load from checkpoint if required
            vn.utils.loadCheckpoint(sess, ckpt_dir)
        except Exception as e:
            print(traceback.print_exc())

        u_op = tf.get_collection('u_op')[0]
        u_var = tf.get_collection('u_var')[0]
        f_var = tf.get_collection('f_var')[0]

        # run the model
        print('start denoising')
        eval_start_time = time.time()
        u_i = sess.run(u_op, feed_dict={u_var: noisy, f_var: noisy})
        time_denoise = time.time() - eval_start_time
        print('denoising of {} image took {:f}s'.format(u_i[0,:,:,0].shape, time_denoise))

        print('saving denoised image to "{}"'.format(output_name))
        toimage(u_i[0, :, :, 0] * 255, cmin=0, cmax=255).save(output_name)



