import datetime
from pathlib import Path

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

_latent_dim = 512  # 128
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=2e-5, beta_1=0.5)

kernel_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
gamma_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

disable_eager_execution()


# ********************************************************************

class ReflectionPadding2D(layers.Layer):

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]
        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(
        kernel_size=3,
        strides=1,
        padding="same",
        use_bias=False,
):
    def block(x):
        filters = x.shape[-1]
        input_tensor = x

        # x = ReflectionPadding2D()(input_tensor)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = LeakyReLU(0.1)(x)

        # x = ReflectionPadding2D()(x)
        x = layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            kernel_initializer=kernel_init,
            padding=padding,
            use_bias=use_bias,
        )(x)
        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)
        x = layers.add([input_tensor, x])

        return x

    return block


# ********************************************************************************

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


# ********************************************************************************

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


def upsampleShuffler(
        filters,
        kernel_size=3,
        padding='same',
        activation=True,
        use_bias=False
):
    def block(x):
        x = Conv2D(
            filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init,
            use_bias=use_bias,
        )(x)

        x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_init)(x)

        if activation:
            x = LeakyReLU(0.1)(x)

        x = PixelShuffler()(x)

        return x

    return block


# ********************************************************************************

def get_resnet_generator(
        filters=64,
        num_downsampling_blocks=2,
        num_residual_blocks=12,
        name='resnet_generator',
):
    input_ = layers.Input(shape=IMAGE_SHAPE)

    x = downsample(64, kernel_size=5, strides=1)(input_)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(filters=filters)(x)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block()(x)

    filters *= 2
    x = upsampleShuffler(filters)(x)
    filters *= 2
    x = upsampleShuffler(filters)(x)

    # Final block
    x = layers.Conv2D(3, (5, 5), padding="same")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(input_, x, name=name)
    model.summary()
    return model


# ********************************************************************************

def get_discriminator(filters=64, name='discriminator'):
    input_ = layers.Input(shape=IMAGE_SHAPE)

    x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_init)(input_)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters

    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(2, 2))(x)
    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(2, 2))(x)
    num_filters *= 2
    x = downsample(filters=num_filters, kernel_size=(4, 4), strides=(1, 1))(x)

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_init)(x)

    model = keras.models.Model(inputs=input_, outputs=x, name=name)
    return model


# ********************************************************************************

lambda_cycle = 10.0
lambda_id = 0.5

disc_A = get_discriminator(name="disc_A")
disc_B = get_discriminator(name="disc_B")

disc_A.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
disc_B.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])

gen_AB = get_resnet_generator(name="gen_AB")
gen_BA = get_resnet_generator(name="gen_BA")

img_A = Input(shape=IMAGE_SHAPE)
img_B = Input(shape=IMAGE_SHAPE)

fake_B = gen_AB(img_A)
fake_A = gen_BA(img_B)

reconstr_A = gen_BA(fake_B)
reconstr_B = gen_AB(fake_A)

same_A = gen_BA(img_A)
same_B = gen_AB(img_B)

disc_A.trainable = False
disc_B.trainable = False

valid_A = disc_A(fake_A)
valid_B = disc_B(fake_B)

# ********************************************************************************

cyclegan = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, same_A, same_B])

cyclegan.compile(
    loss=['mae', 'mae', 'mse', 'mse', 'mse', 'mse'],
    loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
    optimizer=optimizer
)

disc_A.summary()
gen_AB.summary()
cyclegan.summary()

# ********************************************************************************

try:
    cyclegan.load_weights("models/CycleGan/cyclegan_batch.h5")
    print("... load model")
except:
    print("model does not exist")

# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW/validOL")
images_B = get_image_paths("data_train/LU_NEW/validLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

figsize = (20, 20)

# ************************

output_dir = Path('output/GAN/oliwka_laura')
_, ax = plt.subplots(2, 3, figsize=figsize)

inc = 0
i = 0
for source_image in images_A:
    inc = inc + 1

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image_tensor = gen_AB.predict(source_image_tensor)
    predict_image = numpy.clip(predict_image_tensor[0] * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/predicted_img_{i}.jpg".format(i=inc), predict_image)

    reconstructed_image = gen_BA.predict(predict_image_tensor)[0]
    reconstructed_image = numpy.clip(reconstructed_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/reconstructed_img_{i}.jpg".format(i=inc), reconstructed_image)

    source_image = numpy.clip(source_image * 255, 0, 255).astype(numpy.uint8)
    cv2.imwrite(str(output_dir) + "/img_{i}.jpg".format(i=inc), source_image)

    ax[i, 0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    ax[i, 1].imshow(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
    ax[i, 2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    ax[i, 0].set_title("Obraz A")
    ax[i, 1].set_title("Obraz A na obraz B")
    ax[i, 2].set_title("Rekonstrukcja B na A")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
    ax[i, 2].axis("off")

    i = i + 1

plt.savefig(str(output_dir) + "/result.jpg")
plt.show()
plt.close()

# ************************

output_dir = Path('output/GAN/laura_oliwka')
_, ax = plt.subplots(2, 3, figsize=figsize)

inc = 0
i = 0
for source_image in images_B:
    inc = inc + 1

    source_image_tensor = numpy.expand_dims(source_image, 0)
    predict_image_tensor = gen_BA.predict(source_image_tensor)
    predict_image = numpy.clip(predict_image_tensor[0] * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/predicted_img_{i}.jpg".format(i=inc), predict_image)

    reconstructed_image = gen_AB.predict(predict_image_tensor)[0]
    reconstructed_image = numpy.clip(reconstructed_image * 255, 0, 255).astype(numpy.uint8)

    cv2.imwrite(str(output_dir) + "/reconstructed_img_{i}.jpg".format(i=inc), reconstructed_image)

    source_image = numpy.clip(source_image * 255, 0, 255).astype(numpy.uint8)
    cv2.imwrite(str(output_dir) + "/img_{i}.jpg".format(i=inc), source_image)

    ax[i, 0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    ax[i, 1].imshow(cv2.cvtColor(predict_image, cv2.COLOR_BGR2RGB))
    ax[i, 2].imshow(cv2.cvtColor(reconstructed_image, cv2.COLOR_BGR2RGB))
    ax[i, 0].set_title("Obraz B")
    ax[i, 1].set_title("Obraz B na obraz A")
    ax[i, 2].set_title("Rekonstrukcja A na B")
    ax[i, 0].axis("off")
    ax[i, 1].axis("off")
    ax[i, 2].axis("off")

    i = i + 1

plt.savefig(str(output_dir) + "/result.jpg")
plt.show()
plt.close()