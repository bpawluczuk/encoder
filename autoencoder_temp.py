import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************

height = 128
width = 128
chanels = 1

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

# ********************************************************************

def imagetensor(imagedir, width, height):
    for i, im in enumerate(os.listdir(imagedir)):
        image = Image.open(imagedir + im)
        # image = image.convert('L')
        image = image.resize((width, height), Image.ANTIALIAS)
        if i == 0:
            images = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
        else:
            image = np.expand_dims(np.array(image, dtype=float) / 255, axis=0)
            images = np.append(images, image, axis=0)

    return images


# ********************************************************************

def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def upscale(filters):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = UpSampling2D((2, 2))(x)
        return x

    return block


# ********************************************************************

# encoder

def Encoder():
    input_ = Input(shape=IMAGE_SHAPE, name='encoder_input')
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Dense(ENCODER_DIM)(Flatten()(x))
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale(128)(x)
    return Model(input_, x, name="encoder_model")


# decoder

def Decoder():
    input_ = Input(shape=(8, 8, 512), name="decoder_input")
    x = input_
    # x = upscale(1024)(x)
    x = upscale(512)(x)
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x, name="decoder_model")


# ********************************************************************

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

autoencoder_A.summary()

train_dir = "C:\\Sites\\python\\encoder\\dataset\\frames\\bruce\\"
x_train = imagetensor(train_dir, width, height)
x_test = x_train

autoencoder_A.fit(x_train, x_train,
                 epochs=10,
                 batch_size=4,
                 shuffle=True,
                 validation_data=(x_test, x_test),
                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

train_dir = "C:\\Sites\\python\\encoder\\dataset\\frames\\jenifer\\"
x_train = imagetensor(train_dir, width, height)
x_test = x_train

autoencoder_B.fit(x_train, x_train,
                 epochs=10,
                 batch_size=4,
                 shuffle=True,
                 validation_data=(x_test, x_test),
                 callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

# ********************************************************************

#encoder.save_weights("models/encoder.h5")
#decoder_A.save_weights("models/decoder_A.h5")
#decoder_B.save_weights("models/decoder_B.h5")

try:
    encoder.load_weights("models/encoder.h5")
    decoder_A.load_weights("models/decoder_A.h5")
    decoder_B.load_weights("models/decoder_B.h5")
except:
    pass

# ********************************************************************

test_bruce = "C:\\Sites\\python\\encoder\\dataset\\test\\jenifer\\"
x_test = imagetensor(test_bruce, width, height)
decoded_imgs = autoencoder_A.predict(x_test)

n = 5
plt.figure(figsize=(4, 4))
for i in range(1, n + 1):
    # Display original
    ax = plt.subplot(2, n, i)
    plt.imshow(x_test[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + n)
    plt.imshow(decoded_imgs[i].reshape(width, height))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# decoded_imgs = autoencoder_A.predict(x_test)
# cv2.imshow("", decoded_imgs[0])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

