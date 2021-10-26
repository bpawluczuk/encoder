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


def convInOut(filters, kernel_size=4, padding='same', activation=True, dropout_rate=0):
    return conv(
        filters=filters,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        activation=activation,
        dropout_rate=dropout_rate
    )


def downscale(filters, kernel_size=4, strides=2, padding='same', activation=True, dropout_rate=0):
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

        if activation:
            x = LeakyReLU(0.1)(x)

        x = PixelShuffler()(x)

        return x

    return block


def residual_block(input_):
    filters = input_.shape[-1]

    x = conv(filters, strides=1)(input_)
    x = conv(filters, strides=1, activation=False)(x)

    x = layers.add([input_, x])

    return x


def get_discriminator(name="disc"):
    input_ = Input(shape=IMAGE_SHAPE)

    x = convInOut(64, kernel_size=8)(input_)
    x = conv(128, strides=2)(x)
    x = conv(256, strides=2)(x)
    x = conv(512, strides=2)(x)

    x = convInOut(1)(x)

    return Model(input_, x, name=name)


def get_resnet_generator(name="gen"):
    input_ = Input(shape=IMAGE_SHAPE)

    x = convInOut(64, kernel_size=8)(input_)
    x = conv(128, strides=2)(x)
    x = conv(256, strides=2)(x)

    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)
    x = residual_block(x)

    x = upscale(256)(x)
    x = upscale(128)(x)

    x = convInOut(chanels, kernel_size=8, activation=False)(x)
    x = layers.Activation("tanh")(x)

    return Model(input_, x, name=name)


# ********************************************************************************

lambda_cycle = 10.0  # Cycle-consistency loss
lambda_id = 0.1 * lambda_cycle  # Identity loss

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

img_A_id = gen_BA(img_A)
img_B_id = gen_AB(img_B)

disc_A.trainable = False
disc_B.trainable = False

valid_A = disc_A(fake_A)
valid_B = disc_B(fake_B)

# ********************************************************************************

cyclegan = Model(inputs=[img_A, img_B], outputs=[valid_A, valid_B, reconstr_A, reconstr_B, img_A_id, img_B_id])

cyclegan.compile(
    loss=['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
    loss_weights=[1, 1, lambda_cycle, lambda_cycle, lambda_id, lambda_id],
    optimizer=optimizer
)

disc_A.summary()
gen_AB.summary()
cyclegan.summary()

# ********************************************************************

try:
    cyclegan.load_weights("models/CycleGan/cyclegan_batch.h5")
    print("... load model")
except:
    print("model does not exist")


def save_model_weights():
    cyclegan.save_weights("models/CycleGan/cyclegan_batch.h5")
    print("save model weights")


# ********************************************************************************

images_A = get_image_paths("data_test/OL/trainOL")
images_B = get_image_paths("data_test/LU/trainLU")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

batch_size = 2
epochs = 2
dataset_size = len(images_A)
batches = round(dataset_size / batch_size)
sample_interval = 100

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
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
            % (epoch, epochs,
               batch, batches,
               d_loss[0], 100 * d_loss[1],
               g_loss[0],
               numpy.mean(g_loss[1:3]),
               numpy.mean(g_loss[3:5]),
               numpy.mean(g_loss[5:6]),
               elapsed_time))

        if batch % sample_interval == 0:
            save_model_weights()

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
