from pathlib import Path

import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
import tensorflow_addons as tfa

from lib.utils import get_image_paths
from lib.pixel_shuffler import PixelShuffler
from lib.util_face import getFaceAndCoordinates
from lib.seamless_image import seamless_images

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

# ********************************************************************

def convert_one_image(autoencoder, source_image, output_dir, inc):
    assert source_image.shape == (256, 256, 3)

    result = None

    sourceImageFace = getFaceAndCoordinates(source_image, output_dir, inc)

    source_image_tensor = numpy.expand_dims(image, 0)
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

images_A = get_image_paths("data/OL/trainOL")
images_B = get_image_paths("data/LU/trainLU")

# output_dir = Path('output/VAE/oliwka_laura')
# output_dir.mkdir(parents=True, exist_ok=True)
# inc = 0
# for fn in images_B:
#     inc = inc + 1
#     image = cv2.imread(fn)
#     convert_one_image(autoencoder_A, image, output_dir, inc)

output_dir = Path('output/VAE/laura_oliwka')
output_dir.mkdir(parents=True, exist_ok=True)
inc = 0
for fn in images_A:
    inc = inc + 1
    image = cv2.imread(fn)
    convert_one_image(autoencoder_B, image, output_dir, inc)

# key = cv2.waitKey(0)
