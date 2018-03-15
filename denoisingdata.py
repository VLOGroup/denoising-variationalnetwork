import sys
sys.path.append('..')
from vn import VnBasicData
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread, imresize
import random
from skimage import draw


class VnDenoisingData(VnBasicData):
    def __init__(self, config, queue_capacity=10, u_var=None, f_var=None, g_var=None):
        super(VnDenoisingData, self).__init__(queue_capacity=queue_capacity)
        if 'samples' in config.keys():
            samples = config['samples']
        else:
            samples = -1
        self._image_list = [os.path.join(config['train_dir'], f) for f in os.listdir(config['train_dir'])
                            if os.path.isfile(os.path.join(config['train_dir'], f)) and os.path.splitext(f)[1] == '.jpg']
        assert (samples > 0 or samples == -1) and isinstance(samples, int)
        assert config['batch_size'] >= 1 and isinstance(config['batch_size'], int)
        if samples != -1:
            self._image_list = self._image_list[0:samples]
        self._sigma = config['sigma']/255.0
        self._batch_size = config['batch_size']
        self._H = config['H']
        self._W = config['W']
        self._valH = config['valH']
        self._valW = config['valW']
        self.tf_dtype = [tf.float32 for i in range(3)]

        # define inputs
        if u_var == None:
            self.u = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32, name='u')
        else:
            self.u = u_var[0]

        # define constants
        if f_var == None:
            self.constants = {'f': tf.placeholder(shape=(None, None, None, None), dtype=tf.float32, name='f')}
        else:
            self.constants = {'f': f_var[0]}
        # define the target
        if g_var == None:
            self.target = tf.placeholder(shape=(None, None, None, None), dtype=tf.float32, name='g')
        else:
            self.target = g_var[0]

        # generate eval feed_dict for training output
        target = []
        noisy = []
        self._val_img_list = [os.path.join(config['val_dir'], f) for f in os.listdir(config['val_dir'])
                              if os.path.isfile(os.path.join(config['val_dir'], f)) and os.path.splitext(f)[1] == '.jpg']
        self._val_img_list = sorted(self._val_img_list)
        if os.path.exists(config['val_dir'] + '/sigma_25/'):
            self._generate_noise = False
            self._val_img_list_noisy = [os.path.join(config['val_dir']+ '/sigma_25/', f) for f in os.listdir(config['val_dir'] + '/sigma_25/')
                                        if os.path.isfile(os.path.join(config['val_dir'] + '/sigma_25/', f)) and os.path.splitext(f)[1] == '.npy']
            self._val_img_list_noisy = sorted(self._val_img_list_noisy)
        else:
            self._generate_noise = True
        self._current_eval_image = 0

        for n in range(np.minimum(len(self._val_img_list), 10)):
            f = self._val_img_list[n]
            img = imread(f, 'gray').astype(np.float32) / 255.0
            offset_x = 0
            offset_y = 0
            img = img[offset_y:offset_y + self._valH, offset_x:offset_x + self._valW]
            target.append(img)

            if not self._generate_noise:
                f_n = self._val_img_list_noisy[n]
                img_n = np.load(f_n).astype(np.float32) / 255.0
                img_n = img_n[offset_y:offset_y + self._valH, offset_x:offset_x + self._valW]
            else:
                img_n = img + self._sigma * np.random.randn(*img.shape).astype(np.float32)
            noisy.append(img_n)

        target = np.ascontiguousarray(np.asarray(target)[:, :, :, np.newaxis])

        #noisy = target + self._sigma * np.random.randn(*target.shape)
        noisy = np.ascontiguousarray(np.asarray(noisy)[:, :, :, np.newaxis])

        input = noisy.copy()

        self.eval_feed_dict = {self.u: input.astype(np.float32), self.constants['f']: noisy.astype(np.float32),
                               self.target: target.astype(np.float32)}

    def get_feed_dict(self, sess):
        [input, noisy, target] = sess.run(self._batch)
        return {self.u: input, self.constants['f']: noisy, self.target: target}

    def load(self):
        target = []
        for n in range(self._batch_size):
            f = random.choice(self._image_list)
            img = imread(f, 'gray').astype(np.float32) / 255.0
            offset_y = np.random.randint(0, img.shape[0] - self._H)
            offset_x = np.random.randint(0, img.shape[1] - self._W)
            img = img[offset_y:offset_y + self._H, offset_x:offset_x + self._W]
            target.append(img)

        target = np.ascontiguousarray(np.asarray(target)[:, :, :, np.newaxis])
        noisy = target + self._sigma * np.random.randn(*target.shape)
        input = noisy.copy()

        return [input.astype(np.float32), noisy.astype(np.float32), target.astype(np.float32)]

    def num_eval_images(self):
        return len(self._val_img_list)

    def get_eval_feed_dict(self):
        f = self._val_img_list[self._current_eval_image]
        target = imread(f, 'gray').astype(np.float32) / 255.0

        if not self._generate_noise:
            f_n = self._val_img_list_noisy[self._current_eval_image]
            noisy = np.load(f_n).astype(np.float32) / 255.0
        else:
            noisy = target + self._sigma * np.random.randn(*target.shape).astype(np.float32)

        target = np.ascontiguousarray(np.asarray(target)[np.newaxis, :, :, np.newaxis])
        noisy = np.ascontiguousarray(np.asarray(noisy)[np.newaxis, :, :, np.newaxis])

        #noisy = target + self._sigma * np.random.randn(*target.shape)
        input = noisy.copy()

        # update the current evaluation image index
        self._current_eval_image = (self._current_eval_image + 1) % self.num_eval_images()

        return {self.u: input.astype(np.float32),
                self.constants['f']:noisy.astype(np.float32),
                self.target: target.astype(np.float32)}
