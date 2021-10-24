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

from lib.utils import get_image_paths, load_images, stack_images
from lib.training_data import get_training_data
from lib.pixel_shuffler import PixelShuffler

import tensorflow_addons as tfa
import tensorflow_datasets as tfds

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
zoom = 4  # 64*zoom
width = 256
height = 256
_latent_dim = 256  # 128
_variational = 0
chanels = 3
batch_size = 1

IMAGE_SHAPE = (size, size, chanels)
ENCODER_DIM = 1024

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.95)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# ********************************************************************

def conv(filters, kernel_size=4, strides=2):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=kernel_init
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def convDropout(filters, kernel_size=4, strides=2):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            strides=strides,
            padding='same',
            kernel_initializer=kernel_init
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.2)(x)
        return x

    return block


def upscale(filters, kernel_size=4):
    def block(x):
        x = Conv2D(
            filters * 4,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer=kernel_init
        )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
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


def Encoder(input_):
    x = conv(64, strides=2)(input_)
    x = conv(128, strides=2)(x)
    x = conv(128, strides=1)(x)
    x = conv(256, strides=2)(x)
    x = conv(256, strides=1)(x)
    x = convDropout(512, strides=2)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    if not _variational:
        latent_space = Dense(_latent_dim)(x)
    else:
        latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(16 * 16 * 512, activation="relu")(latent_space)
    x = Reshape((16, 16, 512))(x)
    x = upscale(512)(x)

    return Model(input_, x), z_log_sigma, z_mean


def Decoder():
    input_ = Input(shape=(32, 32, 512))
    x = upscale(256)(input_)
    x = upscale(128)(x)
    x = upscale(64)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


# ********************************************************************

x = Input(shape=IMAGE_SHAPE)

encoder, z_log_sigma, z_mean = Encoder(x)
decoder_A = Decoder()
decoder_B = Decoder()

autoencoder_A = Model(x, decoder_A(encoder(x)))
if not _variational:
    autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_A.compile(optimizer=optimizer, loss=vae_loss)

autoencoder_B = Model(x, decoder_B(encoder(x)))
if not _variational:
    autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_B.compile(optimizer=optimizer, loss=vae_loss)

# ********************************************************************

if not _variational:
    history_lost_a_file = 'history/AE/lost_a.txt'
    history_lost_b_file = 'history/AE/lost_b.txt'
else:
    history_lost_a_file = 'history/VAE/lost_a.txt'
    history_lost_b_file = 'history/VAE/lost_b.txt'

# ********************************************************************

try:
    if not _variational:
        encoder.load_weights("models/AE/encoder.h5")
        decoder_A.load_weights("models/AE/decoder_a.h5")
        decoder_B.load_weights("models/AE/decoder_b.h5")
    else:
        encoder.load_weights("models/VAE/encoder.h5")
        decoder_A.load_weights("models/VAE/decoder_a.h5")
        decoder_B.load_weights("models/VAE/decoder_b.h5")

    print("... load models")
except:
    print("models does not exist")


def save_model_weights():
    if not _variational:
        encoder.save_weights("models/AE/encoder.h5")
        decoder_A.save_weights("models/AE/decoder_a.h5")
        decoder_B.save_weights("models/AE/decoder_b.h5")
    else:
        encoder.save_weights("models/VAE/encoder.h5")
        decoder_A.save_weights("models/VAE/decoder_a.h5")
        decoder_B.save_weights("models/VAE/decoder_b.h5")

    print("save model weights")


# ********************************************************************

encoder.summary()
# autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

class Monitor(keras.callbacks.Callback):

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_batch_begin(self, batch, logs=None):

        for i, img in enumerate(train_ol.take(self.num_img)):
            prediction = autoencoder_A.predict(img)[0]
            prediction = ((prediction * 127.5) + 127.5).astype(numpy.uint8)
            img = ((img[0] * 127.5) + 127.5).numpy().astype(numpy.uint8)

            figure = numpy.stack((img, prediction))
            figure = numpy.concatenate(figure, axis=1)

            cv2.imshow("olTolu", figure)

        cv2.waitKey(1)

buffer_size = 1
image_size = (256, 256)
tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


train_ol = tf.keras.preprocessing.image_dataset_from_directory(
    "data_train/OL_NEW",
    validation_split=0.2,
    subset="training",
    seed=1,
    label_mode=None,
    shuffle=buffer_size,
    image_size=image_size,
    batch_size=batch_size,
)

train_ol = (
    train_ol.map(normalize_img, num_parallel_calls=autotune).cache().shuffle(buffer_size)
)

train_lu = tf.keras.preprocessing.image_dataset_from_directory(
    "data_train/LU_NEW",
    validation_split=0.2,
    subset="training",
    seed=1,
    label_mode=None,
    shuffle=buffer_size,
    image_size=image_size,
    batch_size=batch_size,
)

train_lu = (
    train_lu.map(normalize_img, num_parallel_calls=autotune).shuffle(buffer_size)
)

plotter = Monitor()

autoencoder_A.fit(tf.data.Dataset.zip((train_ol, train_lu)), epochs=1, callbacks=plotter)

