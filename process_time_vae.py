import datetime
import cv2
import numpy
import seaborn as sns

sns.set()

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.python.framework.ops import disable_eager_execution

import tensorflow_addons as tfa
from tensorflow.keras import layers

from lib.utils import get_image_paths, load_images, stack_images
from lib.training_data import get_training_data
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

_latent_dim = 256
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=2e-5, beta_1=0.5)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

def downsample(
        filters,
        kernel_size=3,
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
        kernel_size=3,
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


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], _latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var) * epsilon


def vae_loss(input, x_decoded_mean):
    mse_loss = K.mean(keras.losses.mse(input, x_decoded_mean), axis=(1, 2)) * height * width
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return mse_loss + kl_loss


def Encoder(input_, name="Encoder"):
    x = downsample(64, kernel_size=5)(input_)
    x = downsample(128)(x)
    x = downsample(256)(x)
    x = downsample(512)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)

    filters = 512
    x = upsampleShuffler(filters)(x)

    return Model(input_, x, name=name), z_log_sigma, z_mean


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 256))

    x = upsampleShuffler(256)(input_)
    x = upsampleShuffler(128)(x)
    x = upsampleShuffler(64, filter_times=4)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

    return Model(input_, x, name=name)


# ********************************************************************

x = Input(shape=IMAGE_SHAPE)

encoder, z_log_sigma, z_mean = Encoder(x)
decoder_A = Decoder()
decoder_B = Decoder()

# ********************************************************************

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss=vae_loss, metrics=['accuracy'])

autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_B.compile(optimizer=optimizer, loss=vae_loss, metrics=['accuracy'])

# ********************************************************************

try:
    encoder.load_weights("models/VAE/encoder.h5")
    decoder_A.load_weights("models/VAE/decoder_a.h5")
    decoder_B.load_weights("models/VAE/decoder_b.h5")
    print("... load model")
except:
    print("model does not exist")

# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW/testOL")
images_B = get_image_paths("data_train/LU_NEW/testLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

# ************************

start_time = datetime.datetime.now()

for source_image in images_A:

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image_tensor = autoencoder_B.predict(source_image_tensor)
    predict_image = numpy.clip(predict_image_tensor[0] * 255, 0, 255).astype(numpy.uint8)

process_time = datetime.datetime.now() - start_time

print(process_time)