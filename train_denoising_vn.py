import time
import numpy as np
import scipy.misc as scm
import os
import vn

import tensorflow as tf
import argparse

from denoisingdata import VnDenoisingData

import tensorflow.contrib.icg as icg

class VnDenoisingCell(tf.contrib.icg.VnBasicCell):
    def call(self, t, inputs):
        # get the variables
        u = inputs[0]
        u_t_1 = u[t]

        # get the constant, i.e, the noisy image
        f = self._constants['f']

        # extract options
        vmin = self._options['vmin']
        vmax = self._options['vmax']
        pad = self._options['pad']


        # get the parameters
        param_idx = self.time_to_param_index(t)
        # activation function weights
        w1 = self._params['w1'][param_idx]
        # convolution kernels
        k_1 = self._params['k1'][param_idx]

        if self._options['learn_datatermweight']:
            print('learn lambda')
            lambdaa = self._params['lambda'][param_idx]
        else:
            print('fix lambda')
            lambdaa = self._options['datatermweight_init']

        # define the cell
        u_p = tf.pad(u_t_1, [[0, 0], [pad, pad], [pad, pad], [0, 0]], 'REFLECT')
        u_k1 = tf.nn.conv2d(u_p, k_1, [1, 1, 1, 1], 'SAME')
        f1 = tf.contrib.icg.activation_rbf(u_k1, w1, v_min=vmin, v_max=vmax, num_weights=w1.shape[1],
                                           feature_stride=1)
        Ru = tf.nn.conv2d_transpose(f1, k_1, tf.shape(u_p), [1, 1, 1, 1], 'SAME')[:, pad:-pad, pad:-pad, :] / \
             (self._options['num_filter'])

        # data term
        Du = lambdaa * (u_t_1 - f)

        nabla_f_t = Ru + Du
        u_t = u_t_1 - 1./(1 + lambdaa)*nabla_f_t

        return [u_t]


if __name__ == '__main__':
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_config', type=str, default='./configs/training.yaml')
    parser.add_argument('--network_config', type=str, default='./configs/denoising_vn.yaml')
    parser.add_argument('--data_config', type=str, default='./configs/data.yaml')
    parser.add_argument('--global_config', type=str, default='./configs/global.yaml')

    args = parser.parse_args()

    # Load the configs
    network_config, reg_config = tf.contrib.icg.utils.loadYaml(args.network_config, ['network', 'reg'])
    checkpoint_config, optimizer_config = tf.contrib.icg.utils.loadYaml(args.training_config, ['checkpoint_config', 'optimizer_config'])
    data_config = tf.contrib.icg.utils.loadYaml(args.data_config, ['data_config'])
    global_config = tf.contrib.icg.utils.loadYaml(args.global_config, ['global_config'])

    # Tensorflow config
    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = global_config['tf_allow_gpu_growth']

    # define the output locations
    base_name = os.path.basename(args.network_config).split('.')[0]
    suffix = base_name + '_' + time.strftime('%Y-%m-%d--%H-%M-%S')
    vn.setupLogDirs(suffix, args, checkpoint_config)

    # load training samples
    data = VnDenoisingData(data_config, queue_capacity=global_config['data_queue_capacity'])

    # Create a queue runner that will run 4 threads in parallel to enqueue examples.
    qr = tf.train.QueueRunner(data.queue, [data.enqueue_op] * global_config['data_num_threads'])
    # Create a coordinator, launch the queue runner threads.
    coord = tf.train.Coordinator()

    # define parameters (they must be in the params object in order to be optimized)
    params = tf.contrib.icg.utils.Params()
    const_params = tf.contrib.icg.utils.ConstParams()

    vn.paramdefinitions.add_convolution_params(params, const_params, reg_config['filter1'])
    vn.paramdefinitions.add_activation_function_params(params, reg_config['activation1'])
    if network_config['learn_datatermweight']:
        vn.paramdefinitions.add_dataterm_weights(params, network_config)

    # setup the network
    vn_cell = VnDenoisingCell(params=params.get(),
                             const_params=const_params.get(),
                             inputs=[data.u],
                             constants=data.constants,
                             options=network_config)

    denoising_vn = tf.contrib.icg.VariationalNetwork(cell=vn_cell,
                                                     num_stages=network_config['num_stages'],
                                                     parallel_iterations=global_config['parallel_iterations'],
                                                     swap_memory=global_config['swap_memory'])

    # get all images
    u_all = denoising_vn.get_outputs(stage_outputs=True)[0]

    # define loss
    with tf.variable_scope('loss'):
        psnr = tf.reduce_mean(20 * tf.contrib.icg.utils.log10(1.0 / tf.sqrt(tf.reduce_mean((u_all[-1] - data.target) ** 2, axis=(1, 2, 3)))))
        ssim = vn.utils.ssim(u_all[-1], data.target)
        energy = tf.reduce_mean(tf.reduce_sum((u_all[-1] - data.target) ** 2 / 2.0, axis=(1, 2, 3)))

    # add images and energy to summary
    with tf.variable_scope('loss_summary'):
        tf.summary.scalar('energy', energy)
        tf.summary.scalar('psnr', psnr)
        tf.summary.scalar('ssim', ssim)

    tf.summary.image('input', tf.cast(255 * tf.clip_by_value(data.constants['f'], 0, 1), tf.uint8), collections=['images'])
    for i in range(network_config['num_stages']):
        tf.summary.image('u%d' % (i + 1), tf.cast(255 * tf.clip_by_value(u_all[i+1], 0, 1), tf.uint8), max_outputs=10,
                         collections=['images'])
    tf.summary.image('target', tf.cast(255 * data.target, tf.uint8), collections=['images'])

    tf.summary.image('error', u_all[-1] - data.target, collections=['images'], max_outputs=10)

    # define the optimizer
    optimizer = icg.optimizer.StageIPALMOptimizer(network_config['num_stages'], params, energy, optimizer_config)

    with tf.Session(config=tf_config) as sess:
        # initialize the variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # memorize a few ops and placeholders to be used for evaluation
        psnr_op = tf.add_to_collection('psnr_op', psnr)
        ssim_op = tf.add_to_collection('ssim_op', ssim)
        u_all_op = tf.add_to_collection('u_all_op', u_all)
        u_op = tf.add_to_collection('u_op', u_all[-1])
        u_var = tf.add_to_collection('u_var', data.u)
        f_var = tf.add_to_collection('f_var', data.constants['f'])
        g_var = tf.add_to_collection('g_var', data.target)

        # load from checkpoint if required
        saver = tf.train.Saver(max_to_keep=0)
        # initialize enqueuing threads
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)

        # collect the summaries
        epoch_summaries = tf.summary.merge_all()
        image_summaries = tf.summary.merge_all(key='images')
        train_writer = tf.summary.FileWriter(checkpoint_config['log_dir'] + '/' + suffix + '/train/', sess.graph)

        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        try:
            start_time = time.time()
            for epoch in range(0, optimizer_config['max_iter'] + 1):
                if coord.should_stop():
                    break

                # get next mini batch
                feed_dict = data.get_feed_dict(sess=sess)

                # run a single epoch
                optimizer.minimize(sess, epoch, feed_dict)

                if (epoch % checkpoint_config['summary_modulo'] == 0) or epoch == optimizer_config['max_iter']:
                    fd = data.eval_feed_dict
                    summary = sess.run(epoch_summaries,
                                       feed_dict=fd,
                                       options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'step%d' % epoch)
                    train_writer.add_summary(summary, epoch)

                if (epoch % checkpoint_config['save_modulo'] == 0) or epoch == optimizer_config['max_iter']:
                    # update summary
                    fd = data.eval_feed_dict
                    summary = sess.run(image_summaries,
                                       feed_dict=fd,
                                       options=run_options, run_metadata=run_metadata)
                    train_writer.add_run_metadata(run_metadata, 'images%d' % epoch)
                    train_writer.add_summary(summary, epoch)
                    # save variables to checkpoint
                    saver.save(sess, checkpoint_config['log_dir'] + '/' + suffix + '/checkpoints/checkpoint', global_step=epoch)

                # compute the current energy
                e_i = sess.run(energy, feed_dict=feed_dict)
                print("epoch:", epoch, "energy =", e_i)

            print('Elapsed training time:', time.time() - start_time)

            # Evaluate the performance
            print("Evaluating performance")
            eval_output_dir = checkpoint_config['log_dir'] + '/' + suffix + '/test/'
            psnr_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            ssim_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            time_eval = np.zeros((data.num_eval_images(),), dtype=np.float32)
            for i in range(data.num_eval_images()):
                feed_dict = data.get_eval_feed_dict()
                eval_start_time = time.time()
                psnr_i, ssim_i, u_i = sess.run([psnr, ssim, u_all[-1]], feed_dict=feed_dict)
                time_eval[i] = time.time() - eval_start_time
                psnr_eval[i] = psnr_i
                ssim_eval[i] = ssim_i
                print("{:4d}: {:.3f}dB  {:.4f}".format(i, psnr_i, ssim_i))

                # save the output
                scm.toimage(u_i[0, :, :, 0]*255, cmin=0, cmax=255).save(eval_output_dir + "result_{:d}.png".format(i))

            print("-----")
            print(" AVG: {:.3f}dB  {:.4f}".format(np.mean(psnr_eval), np.mean(ssim_eval)))
            print("=====")
            print(" AVG inference time: {:.6f}s".format(np.mean(time_eval)))

            f = open(eval_output_dir + 'results.txt', 'w')
            f.write('{}: {:.3f}dB {:.4f} {:.6f}s'.format(base_name, np.mean(psnr_eval), np.mean(ssim_eval), np.mean(time_eval)))
            f.flush()
            f.close()

        except Exception as e:
            # Report exceptions to the coordinator.
            coord.request_stop(e)
        finally:
            # Terminate as usual. It is innocuous to request stop twice.
            coord.request_stop()
            coord.join(enqueue_threads)
