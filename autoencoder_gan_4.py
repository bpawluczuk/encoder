import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm

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

PIC_DIR = 'data/laura/'

IMAGES_COUNT = 16

LATENT_DIM = 32
CHANNELS = 3

WIDTH = 128
HEIGHT = 128

images = []
for pic_file in tqdm(os.listdir(PIC_DIR)[:IMAGES_COUNT]):
    pic = Image.open(PIC_DIR + pic_file)
    pic.thumbnail((WIDTH, HEIGHT), Image.ANTIALIAS)
    images.append(np.uint8(pic))


# ********************************************************************

def create_generator():
    gen_input = Input(shape=(LATENT_DIM,))

    x = Dense(128 * 16 * 16)(gen_input)
    x = LeakyReLU()(x)
    x = Reshape((16, 16, 128))(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2DTranspose(256, 4, strides=2, padding='same')(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU()(x)
    x = Conv2D(CHANNELS, 7, activation='tanh', padding='same')(x)

    generator = Model(gen_input, x)
    return generator


def create_discriminator():
    disc_input = Input(shape=(HEIGHT, WIDTH, CHANNELS))

    x = Conv2D(256, 3)(disc_input)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(256, 4, strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)

    x = Dense(1, activation='sigmoid')(x)
    discriminator = Model(disc_input, x)

    optimizer = RMSprop(
        lr=.0001,
        clipvalue=1.0,
        decay=1e-8
    )

    discriminator.compile(
        optimizer=optimizer,
        loss='binary_crossentropy'
    )

    return discriminator


generator = create_generator()
discriminator = create_discriminator()
discriminator.trainable = False

gan_input = Input(shape=(LATENT_DIM,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

optimizer = RMSprop(lr=.0001, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=optimizer, loss='binary_crossentropy')

# ********************************************************************

iters = 150000
batch_size = 16

start = 0
d_losses = []
a_losses = []
images_saved = 0

RES_DIR = 'res2'
FILE_PATH = '%s/generated_%d.png'
if not os.path.isdir(RES_DIR):
    os.mkdir(RES_DIR)

CONTROL_SIZE_SQRT = 6
# control_vectors = np.random.normal(size=(CONTROL_SIZE_SQRT ** 2, LATENT_DIM)) / 2
control_vectors = np.random.normal(0.0, 1.0, size=(batch_size, LATENT_DIM))

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

images_A = get_image_paths("data/laura")
images_B = get_image_paths("data/oliwka")
images_A = (load_images(images_A) - 127.5) / 127.5
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

# ********************************************************************
start = 0
for step in range(iters):

    warped_A, target_A = get_training_data(images_A, batch_size)

    start_time = time.time()
    latent_vectors = np.random.normal(0.0, 1.0, size=(batch_size, LATENT_DIM))
    generated = generator.predict(latent_vectors)

    real = images[start:start + batch_size]
    combined_images = np.concatenate([generated, target_A])

    labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
    labels += .05 * np.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)
    d_losses.append(d_loss)

    latent_vectors = np.random.normal(0.0, 1.0, size=(batch_size, LATENT_DIM))
    misleading_targets = np.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(latent_vectors, misleading_targets)
    a_losses.append(a_loss)

    print('%d/%d: d_loss: %.4f,  a_loss: %.4f.  (%.1f sec)' % (step + 1, iters, d_loss, a_loss, time.time() - start_time))

    # control_image = np.zeros((WIDTH * CONTROL_SIZE_SQRT, HEIGHT * CONTROL_SIZE_SQRT, CHANNELS))

    control_generated = generator.predict(control_vectors)

    # for i in range(CONTROL_SIZE_SQRT ** 2):
    #     x_off = i % CONTROL_SIZE_SQRT
    #     y_off = i // CONTROL_SIZE_SQRT
    #     control_image[x_off * WIDTH:(x_off + 1) * WIDTH, y_off * HEIGHT:(y_off + 1) * HEIGHT, :] = control_generated[i,
    #                                                                                                :, :, :]
    # im = Image.fromarray(np.uint8(control_image * 255))
    # im.save(FILE_PATH % (RES_DIR, images_saved))
    # images_saved += 1

    if step % 100 == 0:
        save_model_weights()

    figure_A = np.stack([
        target_A,
        control_generated
    ], axis=1)

    figure = np.concatenate([figure_A], axis=0)
    figure = stack_images(figure)

    figure = np.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

plt.figure(1, figsize=(12, 8))
plt.subplot(121)
plt.plot(d_losses)
plt.xlabel('epochs')
plt.ylabel('discriminant losses')
plt.subplot(122)
plt.plot(a_losses)
plt.xlabel('epochs')
plt.ylabel('adversary losses')
plt.show()
