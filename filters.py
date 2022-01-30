import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import backend as K


def convert_image(file):
    return np.array(Image.open(file).convert('L'))


def my_filter(shape, dtype=None):
    f = np.array([
        [[[-1]], [[-1]], [[-1]]],
        [[[-1]], [[8]], [[-1]]],
        [[[-1]], [[-1]], [[-1]]]
    ])
    return K.variable(f, dtype='float32')


image = convert_image('data_train/OL_NEW/testOL/00000.jpg')
image.shape

plt.imshow(image, cmap='gray')
plt.show()
plt.close()

model = Sequential(
    Conv2D(
        filters=1,
        kernel_size=(3, 3),
        kernel_initializer=my_filter,
        input_shape=(256, 256, 1)
    )
)

model.summary()

image_tensor = tf.expand_dims(image, 0)
result = model.predict(image_tensor)
result = tf.squeeze(result)

plt.imshow(result, cmap='gray')
plt.show()
plt.close()
