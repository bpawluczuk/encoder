import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Conv2D


def convert_image(file):
    return np.array(Image.open(file).convert('L'))


image = convert_image('/Users/bpawluczuk/Sites/python/encoder/data_train/OL_NEW/testOL/00000.jpg')
# image.shape

plt.imshow(image, cmap='gray')
