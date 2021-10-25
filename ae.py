import cv2
import numpy

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa
from tensorflow.python.framework.ops import disable_eager_execution

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

# disable_eager_execution()

buffer_size = 4

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
image_size = (256, 256)
ENCODER_DIM = 1024

adam_optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.95)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

# *********************************************************************

tfds.disable_progress_bar()
autotune = tf.data.experimental.AUTOTUNE


# *********************************************************************

def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


# *********************************************************************

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


# *********************************************************************

def conv(filters, kernel_size=3, strides=2):
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


def convDropout(filters, kernel_size=3, strides=2):
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


def upscale(filters, filter_times=4, kernel_size=3, name=""):
    def block(x):
        x = Conv2D(
            filters * filter_times,
            kernel_size=kernel_size,
            padding='same',
            kernel_initializer=kernel_init,
            name=name
        )(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x

    return block


def Encoder(input_, name="Encoder"):
    x = conv(64, strides=2, kernel_size=7)(input_)
    x = conv(128, strides=2)(x)
    x = conv(128, strides=1)(x)
    x = conv(256, strides=2)(x)
    x = conv(256, strides=1)(x)
    x = convDropout(512, strides=2)(x)
    x = Flatten()(x)

    latent_space = Dense(_latent_dim)(x)

    x = Dense(16 * 16 * 128, activation="relu")(latent_space)
    x = Reshape((16, 16, 128))(x)
    x = upscale(512, filter_times=4)(x)

    encoder = Model(input_, x, name=name)
    encoder.summary()

    return encoder


def Decoder(name="Decoder"):
    input_ = Input(shape=(32, 32, 512))

    x = upscale(256, filter_times=2, name="decoder_up_1")(input_)
    x = upscale(128, filter_times=2, name="decoder_up_2")(x)
    x = upscale(64, filter_times=4, name="decoder_up_3")(x)

    x = Conv2D(3, kernel_size=7, padding='same', activation='sigmoid')(x)

    decoder = Model(input_, x, name=name)
    decoder.summary()

    return decoder


# *********************************************************************

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class AutoEncoder(keras.Model):

    def __init__(self, auto_encoder_A, auto_encoder_B):
        super(AutoEncoder, self).__init__()
        self.auto_encoder_A = auto_encoder_A
        self.auto_encoder_B = auto_encoder_B
        self.optimizer_A = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.optimizer_B = keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
        self.total_loss_tracker_A = keras.metrics.Mean(name="total_loss")
        self.total_loss_tracker_B = keras.metrics.Mean(name="total_loss")

    def train_step(self, data):
        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            reconstruction_A = self.auto_encoder_A(x, training=True)
            total_loss_A = keras.losses.mean_squared_error(x, reconstruction_A)

            reconstruction_B = self.auto_encoder_B(y, training=True)
            total_loss_B = keras.losses.mean_squared_error(y, reconstruction_B)

        gradients_A = tape.gradient(total_loss_A, self.auto_encoder_A.trainable_weights)
        self.optimizer_A.apply_gradients(zip(gradients_A, self.auto_encoder_A.trainable_weights))
        self.total_loss_tracker_A.update_state(total_loss_A)

        gradients_B = tape.gradient(total_loss_B, self.auto_encoder_B.trainable_weights)
        self.optimizer_B.apply_gradients(zip(gradients_B, self.auto_encoder_B.trainable_weights))
        self.total_loss_tracker_B.update_state(total_loss_B)

        return {
            "loss_A": self.total_loss_tracker_A.result(),
            "loss_B": self.total_loss_tracker_B.result()
        }


class Monitor(keras.callbacks.Callback):

    def __init__(self, num_img=4):
        self.num_img = num_img

    def on_batch_begin(self, batch, logs=None):

        for i, img in enumerate(train_ol.take(self.num_img)):
            prediction = self.model.auto_encoder_A(img)[0].numpy()

            prediction = ((prediction * 127.5) + 127.5).astype(numpy.uint8)
            img = ((img[0] * 127.5) + 127.5).numpy().astype(numpy.uint8)

            figure = numpy.stack((img, prediction))
            figure = numpy.concatenate(figure, axis=1)

            cv2.imshow("olTolu", figure)

        for i, img in enumerate(train_lu.take(self.num_img)):
            prediction = self.model.auto_encoder_B(img)[0].numpy()

            prediction = ((prediction * 127.5) + 127.5).astype(numpy.uint8)
            img = ((img[0] * 127.5) + 127.5).numpy().astype(numpy.uint8)

            figure = numpy.stack((img, prediction))
            figure = numpy.concatenate(figure, axis=1)

            cv2.imshow("luTool", figure)

        cv2.waitKey(1)

    # def on_epoch_end(self, epoch, logs=None):
    #     auto_encoder_A.save_weights(("models/AE/ae.h5"))


# *********************************************************************

def get_model():
    x = Input(shape=IMAGE_SHAPE)

    encoder = Encoder(x)

    decoder_A = Decoder()
    auto_encoder_A = Model(x, decoder_A(encoder(x)))

    decoder_B = Decoder()
    auto_encoder_B = Model(x, decoder_B(encoder(x)))

    model = AutoEncoder(auto_encoder_A=auto_encoder_A, auto_encoder_B=auto_encoder_B)

    model.compile(optimizer=Adam(lr=5e-5, beta_1=0.5, beta_2=0.95))

    return model


# *********************************************************************

plotter = Monitor()

auto_encoder_A = get_model()

auto_encoder_A.fit(
    tf.data.Dataset.zip((train_ol, train_lu)),
    validation_data=tf.data.Dataset.zip((train_ol, train_lu)),
    epochs=1,
    callbacks=[plotter]
)
