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

# from matplotlib import pyplot as plt

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except Exception:
    raise ImportError("Pls install imgaug with (pip install imgaug)")

#pls download the images data#
daytime_pictures_dir = 'h:/Data/BDD100K/bdd/images/100k/daytime_img_train'
darktime_pictures_dir = 'h:/Data/BDD100K/bdd/images/100k/daytime_img_val/groundtruth'
real_night_pictures_dir = 'h:/Data/BDD100K/bdd/images/100k/night_img_val'

img_size = (278, 418)

class bdd_daytime(object):
    """a class to load the data from bdd100k

    Example:
        provider_1 = bdd_daytime(batch_size=10, for_what='train', shuffle=True)
        provider_2 = bdd_daytime(batch_size=10, for_what='test', shuffle=True)
        provider_2 = bdd_daytime(batch_size=10, for_what='real_night_test', shuffle=True)

        for i in range(100):
            raw_imgs, dark_imgs = provider_1.load_batch()
            # raw_imgs, dark_imgs = provider_2.load_batch()
            # raw_imgs, dark_imgs = provider_3.load_batch()
            #to do#

    """
    def __init__(self, batch_size, for_what, shuffle):
        assert for_what in ['train', 'test', 'real_night_test']
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
        elif self.for_what == 'test':
            imgs_match = os.path.join(darktime_pictures_dir, '*.jpg')
        elif self.for_what == 'real_night_test':
            imgs_match = os.path.join(real_night_pictures_dir, '*.jpg')
        else:
            raise  ValueError('wrong!!')

        if shuffle:
            self.imgs_full_name = glob.glob(imgs_match)  ##a list
        else:
            self.imgs_full_name = deque(glob.glob(imgs_match))

    def images_number(self):
        return len(self.imgs_full_name)


    def load_batch(self):
        if self.for_what == 'train':
            raw_imgs = []
            dark_imgs= []
            for i in range(self.batch_size):
                raw_img, dark_img = self.load_one_img(aug=True)
                raw_imgs.append(raw_img)
                dark_imgs.append(dark_img)
            return np.array(raw_imgs), np.array(dark_imgs)
        elif self.for_what == 'test':
            raw_imgs = []
            dark_imgs = []
            for i in range(self.batch_size):
                raw_img, dark_img = self.load_one_img()
                raw_imgs.append(raw_img)
                dark_imgs.append(dark_img)
            return np.array(raw_imgs), np.array(dark_imgs)
        elif self.for_what == 'real_night_test':
            raw_imgs = []
            dark_imgs = []
            for i in range(self.batch_size):
                raw_img, dark_img = self.load_one_img()
                raw_imgs.append(raw_img)
                dark_imgs.append(dark_img)
            return np.array(raw_imgs), np.array(dark_imgs)
        else:
            raise ValueError('wrong!!')


    def load_one_img(self, normalize=True, aug=True):
        if self.shuffle:
            file_name = random.sample(self.imgs_full_name, 1)[0]
        else:
            file_name = self.imgs_full_name.popleft()
            self.imgs_full_name.append(file_name)

        if self.for_what == 'train':
            raw_img = self.read_one_img(file_name)

            img = self.reduce_light(raw_img)

            hist = np.random.randint(0, 10, 1)
            if hist >= 4:
                alpha = np.random.randint(10, 50, 1) / 100
                img = transform.match_histograms(img, self.std_dark*alpha)

            if aug:
                seq_det = data_aug_seq.to_deterministic()
                img = seq_det.augment_images([img.astype(np.uint8)])[0]

            dark_img = transform.resize(img, (img_size[0], img_size[1]))
            light_img = transform.resize(raw_img, (img_size[0], img_size[1]))

            n_gaussian = random.randint(5, 10)
            points = []
            radius = []
            for i in range(n_gaussian):
                x = random.randint(0, img_size[1]-1)
                y = random.randint(0, img_size[0]-1)
                radiu = random.uniform(50, max(*img_size)/3)
                radius.append(radiu)
                points.append(np.array([x, y]))

            hm = self.__heat_map(img_size, points, radius)
            hm = np.greater(hm, 0.8).astype(np.float32)
            hm_ = 1. - hm
            dark_img = dark_img*np.expand_dims(hm, axis=-1) + light_img*np.expand_dims(hm_, axis=-1)
            cv2.imshow('hm', hm)

            return (light_img, dark_img) if normalize \
                else ((light_img*255.).astype(np.uint8), (dark_img*255.).astype(np.uint8))  ##normalize to -1 -- 1

        elif self.for_what == 'test':
            raw_img = self.read_one_img(file_name)
            raw_img = cv2.resize(raw_img, (img_size[1], img_size[0]))
            dark_img = self.reduce_light(raw_img)
            belta = np.random.randint(10, 100, 1) / 100
            dark_img = transform.match_histograms(dark_img, self.std_dark * belta)

            return (raw_img*2./255.-1., dark_img*2./255.-1.) if normalize \
                else (raw_img.astype(np.uint8), dark_img.astype(np.uint8))

        elif self.for_what == 'real_night_test':
            raw_img = self.read_one_img(file_name)
            dark_img = cv2.resize(raw_img, (img_size[1], img_size[0]))

            belta = np.random.randint(10, 20, 1) / 100
            dark_img = transform.match_histograms(dark_img, self.std_dark * belta)


            return (raw_img * 2. / 255. - 1., dark_img * 2. / 255. - 1.) if normalize \
                else (raw_img.astype(np.uint8), dark_img.astype(np.uint8))  ##normalize to -1 -- 1

        else:
            raise ValueError('wrong!!')


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
        gamma = np.random.uniform(2, 6, 1)
        alpha = np.random.uniform(0.1, 0.5, 1)

        return (exposure.adjust_gamma(img, gamma)*alpha).astype(np.uint8)

    def __heat_map(self, img_size, points, sigmas):
        """produce a heat map(gray scale) according the points
        Args:
            img_size: img height and width
            points: ndarray or list, represents the coordinate of points. (x, y)
            sigma: control the heap point range
        return:
            a heap map with the shape (h, w)
        Example:
            aa = heat_map(img_size=(224, 224), points=[[50, 50], [100, 100]], sigma=2)
            cv2.imshow('test', aa)
            cv2.waitKey()
            cv2.destroyAllWindows()
        """
        assert self.__legal_points(img_size, points)

        x = np.arange(0, img_size[1], 1)
        y = np.arange(0, img_size[0], 1)
        z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)
        z = np.swapaxes(z, axis1=0, axis2=1)
        heat_map = np.array([self.__gaussian_2d(z, point, sigma=sigma) for point, sigma in zip(points, sigmas)])
        heat_map = np.max(heat_map, axis=0)
        return heat_map / np.max(heat_map)


    def __legal_points(self, img_size, points):
        """judge all points whether in img range
        Args:
            img_size: [img_h, img_w]
            points: a list of point in x,y coordinate.
        Return:
            if all legal, True
            else, False
        """
        for point in points:
            if point[0] < 0 or point[0] > img_size[1] - 1:
                return False
            if point[1] < 0 or point[1] > img_size[0] - 1:
                return False
        return True


    def __gaussian_2d(self, point, mu, sigma):
        """2d gaussion function
        Args:
            point: 2d point coordinate
            mu: 2d mean value
            sigma: the standard deviation in gaussian func
        Return:
            a img with the shape (h, w)
        Example:
            x = np.arange(0, 224, 1)
            y = np.arange(0, 224, 1)
            z = np.swapaxes(np.array(np.meshgrid(x, y)), axis1=0, axis2=2)
            z = gauss_2d(z, mu=(100,100), sigma=2)
            cv2.imshow('test', z)
            cv2.waitKey()
            cv2.destroyAllWindows()
        """
        h = point.shape[0]
        w = point.shape[1]
        point = np.reshape(point, newshape=[-1, 2])
        score = np.exp(-(np.sum(np.square(np.array(point)-np.array(mu)), axis=-1))/(2*sigma**2))

        return np.reshape(score, newshape=[h, w])

## a seq of img augumentation ##
data_aug_seq = iaa.SomeOf(2,[

        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.8,
                      iaa.GaussianBlur(sigma=(1, 3))
                      ),

        # Add gaussian noise.
        # For 50% of all images, we sample the noise once per pixel.
        # For the other 50% of all images, we sample the noise per pixel AND
        # channel. This can change the color (not only brightness) of the
        # pixels.
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.1)]
        , random_order=True)  # apply augmenters in random order


if __name__ == '__main__':
    provider = bdd_daytime(10, 'train', shuffle=True)

    for i in range(100):
        img, dark_img = provider.load_one_img(normalize=False)
        cv2.imshow('light', img)
        cv2.imshow('dark', dark_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        # plt.subplot(121)
        # io.imshow(img.astype(np.uint8))
        # plt.subplot(122)
        # io.imshow(dark_img.astype(np.uint8))
        # plt.show()

    pass
