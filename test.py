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

light_file_match = './light/*.jpg'
light_files = glob.glob(light_file_match)

dark_file_match = './dark/*.jpg'
dark_files = glob.glob(dark_file_match)

img = np.zeros(shape=(720, 1280, 3))
for file in dark_files:
    img += io.imread(file)

img /= len(dark_files)
# io.imshow(img.astype(np.uint8))
# plt.show()

ref_img = img

