import datetime

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

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.95)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

def conv(filters, kernel_size=4, strides=2, padding='same', activation=True, dropout_rate=0):
    def block(x):

        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            kernel_initializer=kernel_init
        )(x)

        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        if dropout_rate:
            x = Dropout(dropout_rate)(x)

        return x

    return block


def convInOut(filters, kernel_size=3, padding='same', activation=True, dropout_rate=0):
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        activation=activation,
        dropout_rate=dropout_rate
    )


def downscale(filters, kernel_size=3, strides=2, padding='same', activation=True, dropout_rate=0):
    return conv(
        filters=filters // 2,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        dropout_rate=dropout_rate
    )


def upscale(filters, kernel_size=4, filter_times=2, padding='same', activation=True):
    def block(x):
        x = Conv2D(
            filters * filter_times,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init
        )(x)
        x = PixelShuffler()(x)
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
    x = conv(64, kernel_size=5)(input_)
    x = conv(64, strides=1)(x)
    x = conv(128)(x)
    x = conv(128, strides=1)(x)
    x = conv(256)(x)
    x = conv(256, strides=1)(x)
    x = conv(512, dropout_rate=0.4)(x)
    x = conv(512, strides=1)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)
    x = upscale(512, filter_times=4)(x)

    return Model(input_, x, name=name), z_log_sigma, z_mean


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 512))

    x = upscale(512)(input_)
    x = conv(512, strides=1)(x)
    x = upscale(256)(x)
    x = conv(256, strides=1)(x)
    x = upscale(128, filter_times=4)(x)
    x = conv(128, strides=1)(x)

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

# encoder.summary()
autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

try:
    encoder.load_weights("models/VAE/encoder.h5")
    decoder_A.load_weights("models/VAE/decoder_a.h5")
    decoder_B.load_weights("models/VAE/decoder_b.h5")
    print("... load model")
except:
    print("model does not exist")


def save_model_weights():
    encoder.save_weights("models/VAE/encoder.h5")
    decoder_A.save_weights("models/VAE/decoder_a.h5")
    decoder_B.save_weights("models/VAE/decoder_b.h5")
    print("save model weights")


# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW_NEW/trainOL")
images_B = get_image_paths("data_train/LU_NEW_NEW/trainLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

batch_size = 1
epochs = 100
dataset_size = len(images_A)
batches = round(dataset_size / batch_size)
save_interval = 110
sample_interval = 5

# ********************************************************************************

start_time = datetime.datetime.now()

for epoch in range(epochs):
    epoch += 1
    for batch in range(batches):
        batch += 1

        warped_A, target_A = get_training_data(images_A, batch_size, size, zoom)
        warped_B, target_B = get_training_data(images_B, batch_size, size, zoom)

        loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
        loss_B = autoencoder_B.train_on_batch(warped_B, target_B)

        loss = 0.5 * numpy.add(loss_A, loss_B)

        elapsed_time = datetime.datetime.now() - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [A loss: %f, acc: %3d%%] [B loss: %f, acc: %3d%%] time: %s " \
            % (epoch, epochs,
               batch, batches,
               loss[0]/10000, 100 * loss[1],
               loss[1], loss[0]/100,
               elapsed_time))

        if batch % save_interval == 0:
            save_model_weights()

        if batch % sample_interval == 0:
            test_A = target_A[0:3]
            test_B = target_B[0:3]

            figure = numpy.stack([
                test_A,
                autoencoder_A.predict(test_A),
                autoencoder_B.predict(test_A),
                test_B,
                autoencoder_B.predict(test_B),
                autoencoder_A.predict(test_B),
            ], axis=1)

            figure = numpy.concatenate([figure], axis=0)
            figure = stack_images(figure)

            figure = numpy.clip(figure * 255, 0, 255).astype(numpy.uint8)

            cv2.imshow("Results", figure)
            key = cv2.waitKey(1)
