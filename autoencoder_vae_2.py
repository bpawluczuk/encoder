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

_shape_before_flattening = 0
_latent_dim = 128


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

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kl_reconstruction_loss(true, pred):
    # Reconstruction loss (binary crossentropy)
    reconstruction_loss = binary_crossentropy(K.flatten(true), K.flatten(pred)) * img_width * img_height

    # KL divergence loss
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    # Total loss = 50% rec + 50% KL divergence loss
    return K.mean(reconstruction_loss + kl_loss)


# Constructing encoder

encoder_input = Input(shape=IMAGE_SHAPE)
x = conv(128)(encoder_input)
x = conv(256)(x)
x = conv(512)(x)
x = conv(1024)(x)
encoder = Flatten()(x)
x = Dense(ENCODER_DIM, activation='relu')(encoder)
x = Dense(8 * 8 * 512, activation='relu')(x)

z_mean = Dense(_latent_dim, name="z_mean")(x)
z_log_var = Dense(_latent_dim, name="z_log_var")(x)

latent_space = Sampling()([z_mean, z_log_var])

# Constructing decoder

decoder_input = Input(shape=(_latent_dim,))
print("decoder_input" + str(K.int_shape(decoder_input)))

x = Dense(8 * 8 * 512, activation="relu")(decoder_input)
x = Reshape((8, 8, 512))(x)

x = upscale(512)(x)
x = upscale(256)(x)
x = upscale(256)(x)
x = upscale(128)(x)

decoder_conv = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

# ********************************************************************

encoder = Model(encoder_input, latent_space)
decoder_A = Model(decoder_input, decoder_conv)
decoder_B = Model(decoder_input, decoder_conv)

vae_A = Model(encoder_input, decoder_A(encoder(encoder_input)))
vae_A.compile(optimizer=optimizer, loss=kl_reconstruction_loss)

vae_B = Model(encoder_input, decoder_B(encoder(encoder_input)))
vae_B.compile(optimizer=optimizer, loss=kl_reconstruction_loss)

# ********************************************************************

encoder.summary()
vae_A.summary()

# ********************************************************************

def save_model_weights():
    encoder.save_weights("models/128/encoder.h5")
    decoder_A.save_weights("models/128/decoder_A.h5")
    decoder_B.save_weights("models/128/decoder_B.h5")
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

    loss_A = vae_A.train_on_batch(warped_A, target_A)
    loss_B = vae_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
        save_model_weights()
        test_A = target_A[0:14]
        test_B = target_B[0:14]

    figure_A = numpy.stack([
        test_A,
        vae_A.predict(test_A),
        vae_B.predict(test_A),
    ], axis=1)

    figure_B = numpy.stack([
        test_B,
        vae_B.predict(test_B),
        vae_A.predict(test_B),
    ], axis=1)

    figure = numpy.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

# ********************************************************************
