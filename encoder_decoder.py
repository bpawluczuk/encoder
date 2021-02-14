import os, shutil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.python.keras.models import load_model

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************

# test_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/test/cloony/"
# train_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/cloony/"

test_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/test/craig/"
train_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/craig/"

# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\cloony\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\cloony\\"
#
# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\craig\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\craig\\"

height = 128
width = 128


# ********************************************************************

def imagetensor(imagedir, width, height):
    for i, im in enumerate(os.listdir(imagedir)):
        image = Image.open(imagedir + im)
        image = image.convert('L')
        image = image.resize((width, height), Image.ANTIALIAS)
        if i == 0:
            images = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
        else:
            image = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
            images = np.append(images, image, axis=0)

    return images


x_train = imagetensor(train_dir, width, height)
x_test = x_train

# ********************************************************************

autoencoder = load_model('encoder_craig_1.h5')
autoencoder.summary()

layer_name = 'conv2d_2'
craig_encoder = Model(autoencoder.input, autoencoder.get_layer(layer_name).output)
# craig_encoder.summary()

# layer_name = 'conv2d_3'
decoder_input = Input(shape=(None, 32, 32, 64))
# x = autoencoder.get_layer(layer_name)(decoder_input)
# x = autoencoder.get_layer('up_sampling2d')(x.output)
# # x = autoencoder.get_layer('conv2d_4')(x)


decoder = Model(decoder_input, autoencoder.get_layer('conv2d_5').output)


decoder.summary()

# ********************************************************************


# x_test = imagetensor(test_dir, width, height)
#
# decoded_img = autoencoder.predict(x_test)
# plt.figure()
# plt.gray()
# plt.imshow(decoded_img.reshape(width, height))
# plt.show()
