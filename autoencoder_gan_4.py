import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import tensorflow as tf

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import RMSprop

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data
from pixel_shuffler import PixelShuffler
import time

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

# ********************************************************************

LATENT_DIM = 128
CHANNELS = 3

WIDTH = 128
HEIGHT = 128

iters = 150000
batch_size = 8

start = 0
d_losses = []
a_losses = []
images_saved = 0

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)


# ********************************************************************

def minimax_discriminator_loss(
        discriminator_real_outputs,
        discriminator_gen_outputs,
        label_smoothing=0.25,
        real_weights=1.0,
        generated_weights=1.0,
        scope=None,
        loss_collection=tf.compat.v1.GraphKeys.LOSSES,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    with tf.compat.v1.name_scope(
            scope, 'discriminator_minimax_loss',
            (discriminator_real_outputs, discriminator_gen_outputs, real_weights,
             generated_weights, label_smoothing)) as scope:
        # -log((1 - label_smoothing) - sigmoid(D(x)))
        loss_on_real = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.ones_like(discriminator_real_outputs),
            discriminator_real_outputs,
            real_weights,
            label_smoothing,
            scope,
            loss_collection=None,
            reduction=reduction)
        # -log(- sigmoid(D(G(x))))
        loss_on_generated = tf.compat.v1.losses.sigmoid_cross_entropy(
            tf.zeros_like(discriminator_gen_outputs),
            discriminator_gen_outputs,
            generated_weights,
            scope=scope,
            loss_collection=None,
            reduction=reduction)

        loss = loss_on_real + loss_on_generated
        tf.compat.v1.losses.add_loss(loss, loss_collection)

        if add_summaries:
            tf.compat.v1.summary.scalar('discriminator_gen_minimax_loss',
                                        loss_on_generated)
            tf.compat.v1.summary.scalar('discriminator_real_minimax_loss',
                                        loss_on_real)
            tf.compat.v1.summary.scalar('discriminator_minimax_loss', loss)

    return loss


def minimax_generator_loss(
        discriminator_gen_outputs,
        label_smoothing=0.0,
        weights=1.0,
        scope=None,
        loss_collection=tf.compat.v1.GraphKeys.LOSSES,
        reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
        add_summaries=False):
    with tf.compat.v1.name_scope(scope, 'generator_minimax_loss') as scope:
        loss = - minimax_discriminator_loss(
            tf.ones_like(discriminator_gen_outputs),
            discriminator_gen_outputs, label_smoothing, weights, weights, scope,
            loss_collection, reduction, add_summaries=False)

    if add_summaries:
        tf.compat.v1.summary.scalar('generator_minimax_loss', loss)

    return loss


# ********************************************************************

def create_generator():
    gen_input = Input(shape=(LATENT_DIM,))

    x = Dense(128 * 128 * 3)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((128, 128, 3))(x)

    x = Conv2D(64, 24, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 12, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 6, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 6, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(CHANNELS, 6, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)

    # generator.compile(
    #     optimizer=optimizer,
    #     loss=minimax_generator_loss
    # )

    return generator


def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(128, 24)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 12, strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 6, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 2, strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)

    discriminator = Model(disc_input, x)

    discriminator.compile(
        optimizer=optimizer,
        loss=minimax_discriminator_loss
    )

    return discriminator


generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

gan.compile(optimizer=optimizer, loss='binary_crossentropy')

# ********************************************************************

control_vectors = np.random.uniform(-1, 1, size=(batch_size, LATENT_DIM))

# ********************************************************************

try:
    gan.load_weights("models/GAN/gan.h5")
    print("... load models")
except:
    print("models does not exist")


def save_model_weights():
    gan.save_weights("models/GAN/gan.h5")
    print("save model weights")


# ********************************************************************

images_A = get_image_paths("dataset/frames/laura_face")
images_A = (load_images(images_A) - 127.5) / 127.5

# ********************************************************************

for step in range(iters):

    warped_A, target_A = get_training_data(images_A, batch_size)

    start_time = time.time()
    latent_vectors = np.random.uniform(-1, 1, size=(batch_size, LATENT_DIM))
    generated = generator.predict(latent_vectors)

    combined_images = np.concatenate([generated, target_A])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.uniform(-1, 1, size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    print(
        '%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

    control_generated = generator.predict(control_vectors)

    if step % 100 == 0:
        save_model_weights()

    figure_A = np.stack([
        target_A,
        control_generated
    ], axis=1)

    figure = np.concatenate([figure_A], axis=0)
    figure = stack_images(figure)

    figure = ((figure * 127.5) + 127.5).astype("uint8")

    cv2.imshow("", figure)
    key = cv2.waitKey(1)
