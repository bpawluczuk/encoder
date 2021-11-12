from pathlib import Path

import cv2
import numpy

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow_addons as tfa
from tensorflow.keras import layers

from lib.utils import get_image_paths, load_images
from lib.pixel_shuffler import PixelShuffler

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

devices = session.list_devices()
for d in devices:
    print(d.name)

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# ********************************************************************

size = 256
width = 256
height = 256
chanels = 3

IMAGE_SHAPE = (size, size, chanels)
ENCODER_DIM = 1024

_latent_dim = 256  # 128
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=2e-5, beta_1=0.5)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

def downsample(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        activation=True,
        dropout_rate=0,
        use_bias=False,
):
    def block(x):
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)
        return x

    return block


def upsampleShuffler(filters, kernel_size=4, filter_times=2, padding='same', activation=True):
    def block(x):
        x = Conv2D(
            filters * filter_times,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init
        )(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        x = PixelShuffler()(x)

        return x

    return block


def upsampleTranspose(
        filters,
        kernel_size=4,
        strides=2,
        padding="same",
        activation=True,
        use_bias=False,
):
    def block(x):
        x = layers.Conv2DTranspose(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_init,
            use_bias=use_bias,
        )(x)

        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        return x

    return block


def Encoder(input_, name="Encoder"):
    x = downsample(64, kernel_size=5)(input_)
    x = downsample(128)(x)
    x = downsample(256)(x)
    x = downsample(512)(x)
    x = Flatten()(x)

    latent_space = Dense(_latent_dim)(x)

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)

    filters = 512
    x = upsampleShuffler(filters)(x)

    return Model(input_, x, name=name)


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 256))

    x = upsampleShuffler(256)(input_)
    x = upsampleShuffler(128)(x)
    x = upsampleShuffler(64, filter_times=4)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

    return Model(input_, x, name=name)


# ********************************************************************

x = Input(shape=IMAGE_SHAPE)

encoder = Encoder(x)
decoder_A = Decoder()
decoder_B = Decoder()

# ********************************************************************

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error', metrics=['accuracy'])

encoder.summary()
autoencoder_A.summary()
autoencoder_B.summary()

# ********************************************************************

try:
    encoder.load_weights("models/AE/encoder.h5")
    decoder_A.load_weights("models/AE/decoder_a.h5")
    decoder_B.load_weights("models/AE/decoder_b.h5")
    print("... load model")
except:
    print("model does not exist")

# *******************************************************************

images_A = get_image_paths("data_train/OL_NEW/testOL")
images_B = get_image_paths("data_train/LU_NEW/testLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

figsize = 20, 20

# ************************

output_dir = Path('output/AE/oliwka_laura')
_, ax = plt.subplots(5, 3, figsize=figsize)

inc = 0
i = 0
for source_image in images_A:
    inc = inc + 1

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image_tensor = autoencoder_B.predict(source_image_tensor)
    predict_image = numpy.clip(predict_image_tensor[0] * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/predicted_img_{i}.jpg".format(i=inc), predict_image)

    reconstructed_image = autoencoder_A.predict(predict_image_tensor)[0]
    reconstructed_image = numpy.clip(reconstructed_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/reconstructed_img_{i}.jpg".format(i=inc), reconstructed_image)

    source_image = numpy.clip(source_image * 255, 0, 255).astype(numpy.uint8)
    cv2.imwrite(str(output_dir) + "/img_{i}.jpg".format(i=inc), source_image)

    ax[i, 0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    ax[i, 1].imshow(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
    ax[i, 2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    ax[i, 0].set_title("Obraz A")
    ax[i, 1].set_title("Obraz A na obraz B")
    ax[i, 2].set_title("Rekonstrukcja B na A")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
    ax[i, 2].axis("off")

    i = i + 1

plt.savefig(str(output_dir) + "/result.jpg")
plt.show()
plt.close()

# ************************

output_dir = Path('output/AE/laura_oliwka')
_, ax = plt.subplots(5, 3, figsize=figsize)

inc = 0
i = 0
for source_image in images_B:
    inc = inc + 1

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image_tensor = autoencoder_A.predict(source_image_tensor)
    predict_image = numpy.clip(predict_image_tensor[0] * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/predicted_img_{i}.jpg".format(i=inc), predict_image)

    reconstructed_image = autoencoder_B.predict(predict_image_tensor)[0]
    reconstructed_image = numpy.clip(reconstructed_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/reconstructed_img_{i}.jpg".format(i=inc), reconstructed_image)

    source_image = numpy.clip(source_image * 255, 0, 255).astype(numpy.uint8)
    cv2.imwrite(str(output_dir) + "/img_{i}.jpg".format(i=inc), source_image)

    ax[i, 0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    ax[i, 1].imshow(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
    ax[i, 2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    ax[i, 0].set_title("Obraz A")
    ax[i, 1].set_title("Obraz A na obraz B")
    ax[i, 2].set_title("Rekonstrukcja B na A")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
    ax[i, 2].axis("off")

    i = i + 1

plt.savefig(str(output_dir) + "/result.jpg")
plt.show()
plt.close()
