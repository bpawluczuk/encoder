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

_latent_dim = 512  # 128
_variational = 0

zoom = 4  # 64*zoom

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.95)

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
        x,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=3,
        strides=1,
        padding="valid",
        gamma_initializer=gamma_init,
        use_bias=False,
):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(
        dim,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.add([input_tensor, x])
    return x


# ********************************************************************************

def downsample(
        x,
        filters,
        activation,
        kernel_initializer=kernel_init,
        kernel_size=3,
        strides=2,
        padding="same",
        gamma_initializer=gamma_init,
        use_bias=False,
        dropout_rate=0
):
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        kernel_initializer=kernel_initializer,
        padding=padding,
        use_bias=use_bias,
    )(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if activation:
        x = activation(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)
    return x


# ********************************************************************************

# def upsample2(
#         x,
#         filters,
#         activation,
#         kernel_size=3,
#         strides=2,
#         padding="same",
#         kernel_initializer=kernel_init,
#         gamma_initializer=gamma_init,
#         use_bias=False,
# ):
#     x = layers.Conv2DTranspose(
#         filters,
#         kernel_size,
#         strides=strides,
#         padding=padding,
#         kernel_initializer=kernel_initializer,
#         use_bias=use_bias,
#     )(x)
#     x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
#     if activation:
#         x = activation(x)
#
#     return x

def upsample(filters, kernel_size=3, filter_times=2, padding='same', activation=True):
    def block(x):
        x = Conv2D(
            filters * filter_times,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=kernel_init
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
        num_residual_blocks=9,
        num_upsample_blocks=2,
        gamma_initializer=gamma_init,
        name=None,
):
    img_input = layers.Input(shape=IMAGE_SHAPE, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation("relu")(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters, activation=layers.Activation("relu"), dropout_rate=0.4)

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation("relu"))

    # Upsampling
    for _ in range(num_upsample_blocks):
        # filters //= 2
        # x = upsample(x, filters, activation=layers.Activation("relu"))
        x = upsample(filters)(x)

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding="valid")(x)
    x = layers.Activation("tanh")(x)

    model = keras.models.Model(img_input, x, name=name)
    return model


# ********************************************************************************

def get_discriminator(filters=64, kernel_initializer=kernel_init, name=None):
    img_input = layers.Input(shape=IMAGE_SHAPE, name=name + "_img_input")
    x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters

    for num_downsample_block in range(3):

        num_filters *= 2
        if num_downsample_block < 2:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

    model = keras.models.Model(inputs=img_input, outputs=x, name=name)
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

# ********************************************************************

try:
    cyclegan.load_weights("models/CycleGan/cyclegan_test_2.h5")
    print("... load model")
except:
    print("model does not exist")


def save_model_weights():
    cyclegan.save_weights("models/CycleGan/cyclegan_test_2.h5")
    print("save model weights")


# ********************************************************************************

images_A = get_image_paths("data_train/OL_NEW_NEW/trainOL")
images_B = get_image_paths("data_train/LU_NEW_NEW/trainLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

batch_size = 1
epochs = 2000
dataset_size = len(images_A)
batches = round(dataset_size / batch_size)
save_interval = 110
sample_interval = 5

# ********************************************************************************

start_time = datetime.datetime.now()

valid = numpy.ones((batch_size,) + (32, 32, 1))
fake = numpy.zeros((batch_size,) + (32, 32, 1))

for epoch in range(epochs):
    epoch += 1
    for batch in range(batches):
        batch += 1

        warped_A, target_A = get_training_data(images_A, batch_size, size, zoom)
        warped_B, target_B = get_training_data(images_B, batch_size, size, zoom)

        fake_B = gen_AB.predict(target_A)
        fake_A = gen_BA.predict(target_B)

        dA_loss_real = disc_A.train_on_batch(target_A, valid)
        dA_loss_fake = disc_A.train_on_batch(fake_A, fake)
        dA_loss = 0.5 * numpy.add(dA_loss_real, dA_loss_fake)

        dB_loss_real = disc_B.train_on_batch(target_B, valid)
        dB_loss_fake = disc_B.train_on_batch(fake_B, fake)
        dB_loss = 0.5 * numpy.add(dB_loss_real, dB_loss_fake)

        # Total disciminator loss
        d_loss = 0.5 * numpy.add(dA_loss, dB_loss)

        # Train the generators
        g_loss = cyclegan.train_on_batch([target_A, target_B], [valid, valid, target_A, target_B, target_A, target_B])

        elapsed_time = datetime.datetime.now() - start_time

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, same: %05f] time: %s " \
            % (epoch, epochs,
               batch, batches,
               d_loss[0], 100 * d_loss[1],
               g_loss[0],
               numpy.mean(g_loss[1:3]),
               numpy.mean(g_loss[3:5]),
               numpy.mean(g_loss[5:6]),
               elapsed_time))

        if batch % save_interval == 0:
            save_model_weights()

        if batch % sample_interval == 0:
            test_A = target_A[0:3]
            test_B = target_B[0:3]

            fake_B = gen_AB.predict(test_A)
            fake_A = gen_BA.predict(test_B)

            reconstr_A = gen_BA.predict(fake_B)
            reconstr_B = gen_AB.predict(fake_A)

            figure = numpy.stack([
                test_A,
                fake_B,
                reconstr_A,
                test_B,
                fake_A,
                reconstr_B,
            ], axis=1)

            figure = numpy.concatenate([figure], axis=0)
            figure = stack_images(figure)

            figure = numpy.clip(figure * 255, 0, 255).astype(numpy.uint8)

            cv2.imshow("Results", figure)
            key = cv2.waitKey(1)
