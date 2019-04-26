"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import os
import glob
import random
from collections import deque
from skimage import io, exposure, transform
import numpy as np
import cv2

from matplotlib import pyplot as plt

daytime_pictures_dir = 'h:/Data/BDD100K/bdd/images/100k/daytime_img'
darktime_pictures_dir = 'f:/my_project/dark_aug/dark_1'

img_size = (139, 209)

class bdd_daytime(object):
    def __init__(self, batch_size, for_what, shuffle):
        assert for_what in ['train', 'test']
        assert batch_size > 0
        assert type(shuffle) == bool

        self.batch_size = batch_size
        self.for_what = for_what
        self.shuffle = shuffle

        file = os.path.join(os.path.dirname(__file__), 'std_dark.jpg')
        self.std_dark = io.imread(file)
        self.std_dark = cv2.resize(self.std_dark, (img_size[1], img_size[0]))

        if for_what == 'train':
            imgs_match = os.path.join(daytime_pictures_dir, '*.jpg')
            if shuffle:
                self.imgs_full_name = glob.glob(imgs_match) ##a list
            else:
                self.imgs_full_name = deque(glob.glob(imgs_match))
            pass
        else:
            imgs_match = os.path.join(darktime_pictures_dir, '*.jpg')
            if shuffle:
                self.imgs_full_name = glob.glob(imgs_match)  ##a list
            else:
                self.imgs_full_name = deque(glob.glob(imgs_match))
            pass

    def load_batch(self):
        if self.for_what == 'train':
            raw_imgs = []
            train_imgs= []
            for i in range(self.batch_size):
                raw_img, train_img = self.load_ong_img()
                raw_imgs.append(raw_img)
                train_imgs.append(train_img)
            return np.array(train_imgs), np.array(raw_imgs)
        else:
            dark_imgs = []
            input_imgs = []
            for i in range(self.batch_size):
                dark_img, input_img = self.load_ong_img()
                dark_imgs.append(dark_img)
                input_imgs.append(input_img)
            return np.array(dark_imgs), np.array(input_imgs)


    def load_ong_img(self, normalize=True, hist=True, aug=True):
        if self.shuffle:
            file_name = random.sample(self.imgs_full_name, 1)[0]
        else:
            file_name = self.imgs_full_name.popleft()
            self.imgs_full_name.append(file_name)

        if self.for_what == 'train':
            raw_img = self.read_one_img(file_name)
            img = self.reduce_light(raw_img)

            if hist:
                alpha = np.random.randint(10, 50, 1) / 100
                img = transform.match_histograms(img, self.std_dark*alpha)

            img = cv2.resize(img, (img_size[1], img_size[0]))
            raw_img = cv2.resize(raw_img, (img_size[1], img_size[0])).astype(np.float64)

            if aug:
                angle = np.random.randint(-10, 10, 1)
                raw_img = transform.rotate(raw_img, angle)
                img = transform.rotate(img, angle)

            return (raw_img*2./255.-1., img*2./255. - 1.) if normalize \
                else (raw_img.astype(np.uint8), img.astype(np.uint8))  ##normalize to -1 -- 1
        else:
            dark_img = self.read_one_img(file_name)
            dark_img = cv2.resize(dark_img, (img_size[1], img_size[0]))

            alpha = np.random.randint(30, 40, 1) / 100
            input_img = transform.match_histograms(dark_img, self.std_dark * alpha)

            return (dark_img * 2. / 255. - 1.,input_img * 2. / 255. - 1.) if normalize \
                else (dark_img.astype(np.uint8),input_img.astype(np.uint8))  ##normalize to -1 -- 1


    def read_one_img(self, img_name):
        """read one img
        Args:
            img_name: img full name
        Return:
            ndarray of img
        """
        assert os.path.isfile(img_name)

        img = io.imread(img_name)
        return img


    def reduce_light(self, img):
        """random reduce the light of img
        Args:
            img: ndarray of img
        Return:
            ndarray of img after reducing the light
        """
        gamma = np.random.randint(20,50,1)/10
        alpha = np.random.randint(10,50,1)/100
        return (exposure.adjust_gamma(img, gamma)*alpha).astype(np.uint8)



if __name__ == '__main__':
    provider = bdd_daytime(10, 'train', shuffle=True)

    for i in range(100):
        img, sss = provider.load_ong_img(normalize=True)
        img = (img+1.)*255/2
        io.imshow(img.astype(np.uint8))
        plt.show()

    pass
