import datetime
from pathlib import Path

import cv2
import numpy

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

from lib.seamless_image import seamless_images
from lib.util_face import getFaceAndCoordinates
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
    # x = downsample(128, strides=1)(x)
    x = downsample(128)(x)
    # x = downsample(512, strides=1)(x)
    x = downsample(256)(x)
    # x = downsample(512, strides=1)(x)
    x = downsample(512)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)

    filters = 512
    x = upsampleShuffler(filters)(x)
    # x = upsampleShuffler(512, filter_times=4)(x)

    return Model(input_, x, name=name), z_log_sigma, z_mean


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 256))

    # filters = 512
    # filters //= 2
    # x = upsampleTranspose(filters)(input_)
    # x = downsample(filters, strides=1)(x)
    x = upsampleShuffler(256)(input_)

    # filters //= 2
    # x = upsampleTranspose(filters)(x)
    # x = downsample(filters, strides=1)(x)
    x = upsampleShuffler(128)(x)

    # filters //= 2
    # x = upsampleTranspose(filters)(x)
    # x = downsample(filters, strides=1)(x)
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

encoder.summary()
# autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

try:
    encoder.load_weights("models/VAE/encoder.h5")
    decoder_A.load_weights("models/VAE/decoder_a.h5")
    decoder_B.load_weights("models/VAE/decoder_b.h5")
    print("... load model")
except:
    print("model does not exist")


# ********************************************************************

def convert_one_image(autoencoder, source_image, output_dir, inc):
    assert source_image.shape == (256, 256, 3)

    result = None

    sourceImageFace = getFaceAndCoordinates(source_image, output_dir, inc)

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image = autoencoder.predict(source_image_tensor)[0]
    predict_image = numpy.clip(predict_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/" + str(inc) + "_source_image.jpg", source_image)
    cv2.imwrite(str(output_dir) + "/" + str(inc) + "_predict_image.jpg", predict_image)

    if sourceImageFace is not None:
        xmin, ymin, xmax, ymax, h, w, face = sourceImageFace

        source_image_face = cv2.resize(face, (int(128), int(128)))

        # cv2.imshow("Source face", source_image_face)
        # cv2.imwrite(str(output_dir) + "/" + str(inc) + "_source_image_face.jpg", source_image_face)

        destination_image = source_image.copy()
        destination_image[ymin:ymin + h, xmin:xmin + w] = predict_image[ymin:ymin + h, xmin:xmin + w]

        # cv2.imshow("Dest image", destination_image)
        # cv2.imwrite(str(output_dir) + "/" + str(inc) + "_dest_image.jpg", destination_image)

        seamless_destination_image = seamless_images(destination_image, source_image)

        # cv2.imshow("Dest image seamless", seamless_destination_image)
        cv2.imwrite(str(output_dir) + "/" + str(inc) + "_dest_image_seamless.jpg", seamless_destination_image)

        # cv2.imshow("#1", source_image_tensor[0])
        # cv2.imshow("#2", predict_image)

        result = seamless_destination_image

    return result


# *******************************************************************

images_A = get_image_paths("output/resultOL")
images_B = get_image_paths("output/resultLU")

output_dir = Path('output/VAE/laura_oliwka')

inc = 0
for fn in images_A:
    inc = inc + 1
    source_image = cv2.imread(fn)
    cv2.imwrite(str(output_dir) + "/img_{i}.jpg".format(i=inc), source_image)

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image = autoencoder_B.predict(source_image_tensor)[0]
    predict_image = numpy.clip(predict_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/predicted_img_{i}.jpg".format(i=inc), predict_image)

    # convert_one_image(autoencoder_B, source_image, output_dir, inc)

    # key = cv2.waitKey(0)
