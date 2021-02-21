import os
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
import dlib

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************

test_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/test/cloony/"
train_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/cloony/"

# test_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/test/craig/"
# train_dir = "/Users/bpawluczuk/Sites/python/encoder/dataset/train/craig/"

# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\cloony\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\cloony\\"
#
# train_dir = "C:\\Sites\\python\\encoder\\dataset\\train\\craig\\"
# test_dir = "C:\\Sites\\python\\encoder\\dataset\\test\\craig\\"

height = 128
width = 128
chanels = 1


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

encoder_A = load_model('models/encoder_A.h5')
decoder_A = load_model('models/decoder_A.h5')

encoder_B = load_model('models/encoder_B.h5')
decoder_B = load_model('models/decoder_B.h5')

autoencoder_input = Input(shape=(width, height, chanels), name="autoencoder_input")
autoencoder_encoder_output = encoder_A(autoencoder_input)
autoencoder_decoder_output = decoder_B(autoencoder_encoder_output)
autoencoder = Model(autoencoder_input, autoencoder_decoder_output, name="autoencoder")
autoencoder.summary()
#
autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

# ********************************************************************


x_test = imagetensor(test_dir, width, height)

decoded_img = autoencoder.predict(x_test)
plt.figure()
plt.gray()
plt.imshow(decoded_img.reshape(width, height))
plt.show()
