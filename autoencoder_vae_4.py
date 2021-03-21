import cv2
import numpy

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import binary_crossentropy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
from pixel_shuffler import PixelShuffler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# ********************************************************************

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024
img_width = 128
img_height = 128

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

latent_dim = 1024
variational = 0
batch_size = 16


# ********************************************************************


def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.4)(x)
        return x

    return block


def upscale(filters):
    def block(x):
        x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        x = PixelShuffler()(x)
        return x

    return block


# Constructing encoder

def Encoder():
    _input = Input(shape=IMAGE_SHAPE)

    x = conv(128)(_input)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Flatten()(x)

    if not variational:
        latent_space = Dense(latent_dim)(x)
    else:
        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=1.)
            return z_mean + K.exp(z_log_sigma) * epsilon

        z_mean = Dense(latent_dim)(x)
        z_log_sigma = Dense(latent_dim)(x)
        latent_space = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    x = Dense(8 * 8 * 1024, activation="relu")(latent_space)
    x = Reshape((8, 8, 1024))(x)
    encoder_output = upscale(512)(x)
    return Model(_input, encoder_output)


# Constructing decoder

def Decoder():
    decoder_input = Input(shape=(16, 16, 512))

    x = upscale(512)(decoder_input)
    x = upscale(256)(x)
    x = upscale(128)(x)

    decoder_output = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(decoder_input, decoder_output)


# ********************************************************************

encoder_input = Input(shape=IMAGE_SHAPE)

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

autoencoder_A = Model(encoder_input, decoder_A(encoder(encoder_input)))
if not variational:
    autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')

autoencoder_B = Model(encoder_input, decoder_B(encoder(encoder_input)))
if not variational:
    autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

# ********************************************************************

encoder.summary()
autoencoder_A.summary()
autoencoder_B.summary()


# ********************************************************************

def save_model_weights():
    # encoder.save_weights("models/128/encoder.h5")
    # decoder_A.save_weights("models/128/decoder_A.h5")
    # decoder_B.save_weights("models/128/decoder_B.h5")
    print("save model weights")


# try:
#     encoder.load_weights("models/128/encoder.h5")
#     decoder_A.load_weights("models/128/decoder_A.h5")
#     decoder_B.load_weights("models/128/decoder_B.h5")
#     print("... load models")
# except:
#     print("models does not exist")

# ********************************************************************

images_A = get_image_paths("dataset/frames/harrison_face")
images_B = get_image_paths("dataset/frames/ryan_face")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

for epoch in range(100000):
    batch_size = 16
    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
        save_model_weights()
        test_A = target_A[0:14]
        test_B = target_B[0:14]

    figure_A = numpy.stack([
        test_A,
        autoencoder_A.predict(test_A),
        autoencoder_B.predict(test_A),
    ], axis=1)

    figure_B = numpy.stack([
        test_B,
        autoencoder_B.predict(test_B),
        autoencoder_A.predict(test_B),
    ], axis=1)

    figure = numpy.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

# ********************************************************************
