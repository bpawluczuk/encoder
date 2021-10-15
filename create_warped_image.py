import cv2
import numpy

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

from lib.image_augmentation import random_transform, random_warp
from lib.utils import get_image_paths, load_images, stack_images
from lib.training_data import get_training_data
from lib.pixel_shuffler import PixelShuffler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# ********************************************************************

size = 256
zoom = 4  # 64*zoom

# ********************************************************************

images_A = get_image_paths("data_test/OL/trainOL")
images_B = get_image_paths("data_test/LU/trainLU")


random_transform_args = {
    'rotation_range': 10,
    'zoom_range': 0.05,
    'shift_range': 0.05,
    'random_flip': 0.4,
}

inc = 0
for path in images_A:
    inc = inc + 1
    image = cv2.imread(path)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp(image, size, 5, zoom)
    cv2.imwrite("data_train/OL/trainOL/{i}_ol.jpg".format(i=inc), target_img)
    cv2.imwrite("data_train/OL_WARP/trainOL/{i}_ol.jpg".format(i=inc), warped_img)

inc = 0
for path in images_B:
    inc = inc + 1
    image = cv2.imread(path)
    image = random_transform(image, **random_transform_args)
    warped_img, target_img = random_warp(image, size, 5, zoom)
    cv2.imwrite("data_train/LU/trainLU/{i}_lu.jpg".format(i=inc), target_img)
    cv2.imwrite("data_train/LU_WARP/trainLU/{i}_lu.jpg".format(i=inc), warped_img)


# ********************************************************************
