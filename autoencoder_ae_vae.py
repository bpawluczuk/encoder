import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

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

size = 256
zoom = 4  # 64*zoom
width = 256
height = 256
_latent_dim = 256
_variational = 1
chanels = 3

IMAGE_SHAPE = (size, size, chanels)
ENCODER_DIM = 1024

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)


# ********************************************************************

def conv(filters):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def convDropout(filters):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=4,
            strides=2,
            padding='same',
            kernel_initializer=kernel_init
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.4)(x)
        return x

    return block


def upscale(filters):
    def block(x):
        x = Conv2D(
            filters * 4,
            kernel_size=3,
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
    x = conv(128)(input_)
    x = conv(256)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = convDropout(512)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    if not _variational:
        latent_space = Dense(_latent_dim)(x)
    else:
        latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(8 * 8 * 512, activation="relu")(latent_space)
    x = Reshape((8, 8, 512))(x)
    x = upscale(512)(x)

    return Model(input_, x), z_log_sigma, z_mean


def Decoder():
    input_ = Input(shape=(16, 16, 512))

    x = upscale(512)(input_)
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(128)(x)

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

# encoder.summary()
# autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

images_A = get_image_paths("data/oliwka_256/oliwka_256")
images_B = get_image_paths("data/laura_256/laura_256")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

for epoch in range(100000):
    batch_size = 8
    warped_A, target_A = get_training_data(images_A, batch_size, size, zoom)
    warped_B, target_B = get_training_data(images_B, batch_size, size, zoom)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
        save_model_weights()
        test_A = target_A[0:7]
        test_B = target_B[0:7]

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

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

# ********************************************************************
