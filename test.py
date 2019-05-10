"""
Copyright (c) College of Mechatronics and Control Engineering, Shenzhen University.
All rights reserved.

Description :


Author：Team Li
"""
import glob
import os
from skimage import io, exposure, color, transform
from matplotlib import pyplot as plt
import numpy as np
import cv2

darktime_pictures_dir = 'h:/Data/BDD100K/bdd/images/100k/daytime_img_val/groundtruth'

match = os.path.join(darktime_pictures_dir, '*.jpg')
files = glob.glob(match)

std_dark = cv2.imread('./dataset/std_dark.jpg')

for file in files:
    img = cv2.imread(file)
    img = cv2.resize(img, (418, 278))

    img_1 = exposure.adjust_gamma(img, gamma=2)
    # img_2 = exposure.adjust_gamma(img, gamma=3)
    img_3 = exposure.adjust_gamma(img, gamma=4)
    # img_4 = exposure.adjust_gamma(img, gamma=5)
    # img_5 = exposure.adjust_gamma(img, gamma=6)

    # img_1_0 = (img_1 * 0.1).astype(np.uint8)
    # img_1_1 = (img_1 * 0.4).astype(np.uint8)
    # img_1_2 = (img_1 * 0.7).astype(np.uint8)
    #
    # img_2_0 = (img_3 * 0.1).astype(np.uint8)
    # img_2_1 = (img_3 * 0.4).astype(np.uint8)
    # img_2_2 = (img_3 * 0.7).astype(np.uint8)

    img = (img_1 * 0.7).astype(np.uint8)

    img_1_0 = transform.match_histograms(img, 0.1*std_dark)

    img_1_2 = transform.match_histograms(img, 0.3*std_dark)

    img_2_2 = transform.match_histograms(img, 1.0*std_dark)


    # cv2.imshow('raw', img)
    # cv2.imshow('gamma=2', img_1)
    # cv2.imshow('gamma=3', img_2)
    # cv2.imshow('gamma=4', img_3)
    # cv2.imshow('gamma=5', img_4)
    # cv2.imshow('gamma=6', img_5)

    cv2.imshow('1_0', img_1_0.astype(np.uint8))

    cv2.imshow('1_2', img_1_2.astype(np.uint8))


    cv2.imshow('2_2', img_2_2.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
