import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

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

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

_image_shape = (128, 128, 3)
_latent_dim = 256
_batch_size = 16
_variational = 1
width = 128
height = 128


def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
        return x

    return block


def convDropout(filters):
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

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


def Generator(input_):
    x = Dense(128 * 128 * 128)(input_)
    x = LeakyReLU(0.1)(x)
    x = Reshape((128, 128, 2097152))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(3, 7, activation='tanh', padding='same')(x)

    return Model(input_, x)


def Discriminator(input_):
    x = Conv2D(128, 3)(input_)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dropout(0.4)(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(input_, x)


# class CustomVariationalLayer(keras.layers.Layer):
#
#     def vae_loss(self, x, z_decoded):
#         x = K.flatten(x)
#         z_decoded = K.flatten(z_decoded)
#         xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
#         kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#         return K.mean(xent_loss + kl_loss)
#
#     def call(self, inputs):
#         x = inputs[0]
#         z_decoded = inputs[1]
#         loss = self.vae_loss(x, z_decoded)
#         self.add_loss(loss, inputs=inputs)
#         return x


# ********************************************************************

latent_dim = 128

gan_input = keras.Input(shape=(latent_dim,))
generator = Generator(gan_input)
generator.summary()

discriminator_input = Input(shape=IMAGE_SHAPE)
discriminator = Discriminator(discriminator_input)
discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = keras.Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')


# gan_output = Discriminator(autoencoder_A(gan_input))
# gan = Model(gan_input, gan_output)
#
# gan.compile(optimizer=optimizer, loss='mean_absolute_error')
# gan.summary()

# ********************************************************************

batch_size = 16

valid = numpy.ones((batch_size, 1))
fake = numpy.zeros((batch_size, 1))

# ********************************************************************

images_A = get_image_paths("data/laura")
images_B = get_image_paths("data/oliwka")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

# for epoch in range(10):
#     batch_size = 16
#     warped_A, target_A = get_training_data(images_A, batch_size)
#     warped_B, target_B = get_training_data(images_B, batch_size)
#
#     loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
#     loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
#     print(epoch, loss_A, loss_B)
#
#     if epoch % 100 == 0:
#         save_model_weights()
#         test_A = target_A[0:14]
#         test_B = target_B[0:14]
#
#     figure_A = numpy.stack([
#         test_A,
#         autoencoder_A.predict(test_A),
#         autoencoder_B.predict(test_A),
#     ], axis=1)
#
#     figure_B = numpy.stack([
#         test_B,
#         autoencoder_B.predict(test_B),
#         autoencoder_A.predict(test_B),
#     ], axis=1)
#
#     figure = numpy.concatenate([figure_A, figure_B], axis=0)
#     figure = figure.reshape((4, 7) + figure.shape[1:])
#     figure = stack_images(figure)
#
#     figure = numpy.clip(figure * 255, 0, 255).astype('uint8')
#
#     cv2.imshow("", figure)
#     key = cv2.waitKey(1)

# ********************************************************************
