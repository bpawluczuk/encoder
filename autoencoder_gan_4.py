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

latent_dim = 128

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)


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


# ********************************************************************

def Encoder(input_):
    x = conv(128)(input_)
    x = conv(256)(x)
    x = conv(512)(x)
    x = convDropout(512)(x)
    x = Flatten()(x)

    latent_space = Dense(latent_dim)(x)

    x = Dense(8 * 8 * 512, activation="relu")(latent_space)
    x = Reshape((8, 8, 512))(x)
    x = upscale(512)(x)

    return Model(input_, x)


def Decoder():

    input_ = Input(shape=(16, 16, 512))

    x = upscale(512)(input_)
    x = upscale(256)(x)
    x = upscale(128)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

    return Model(input_, x)


def Generator(input_):
    x = Conv2D(128, 5, padding='same')(input_)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256, 5, padding='same')(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128, 5, padding='same')(x)
    # x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    # x = Dropout(0.4)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

    return Model(input_, x)


# ********************************************************************


def Discriminator(input_):
    x = Conv2D(128, kernel_size=5, padding='same')(input_)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=5, padding='same', strides=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=5, padding='same', strides=2)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(256, kernel_size=5, padding='same', strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)

    x = Dropout(0.4)(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(input_, x)


# ********************************************************************

X = Input(shape=IMAGE_SHAPE)

E = Encoder(X)
E.compile(optimizer=optimizer, loss='mean_absolute_error')

G = Decoder()
G.compile(optimizer=optimizer, loss='mean_absolute_error')

D_true = Discriminator(X)
D_true.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
D_true.trainable = False

D_fromGen = Discriminator(X)
D_fromGen.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
D_fromGen.trainable = False

D_fake = Discriminator(X)
D_fake.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
D_fake.trainable = False


AE = Model(X, G(E(X)))
AE.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# ********************************************************************

batch_size = 16

valid = numpy.ones((batch_size, 1))
fake = numpy.zeros((batch_size, 1))

# ********************************************************************

# try:
#     gan.load_weights("models/GAN/gan.h5")
#     print("... load models")
# except:
#     print("models does not exist")
#
#
# def save_model_weights():
#     gan.save_weights("models/GAN/gan.h5")
#     print("save model weights")


# ********************************************************************

images_A = get_image_paths("data/laura")
images_B = get_image_paths("data/oliwka")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

noise = numpy.random.normal(0, 1, (batch_size, 128))

for epoch in range(10000000):
    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    latent_vect = E.predict(target_A)
    encImg = G.predict(latent_vect)
    fakeImg = G.predict(noise)


    DlossTrue = D_true.train_on_batch(target_A, numpy.ones((batch_size, 1)))
    DlossEnc = D_fromGen.train_on_batch(encImg, numpy.ones((batch_size, 1)))
    DlossFake = D_fake.train_on_batch(fakeImg, numpy.zeros((batch_size, 1)))

    # GlossEnc = G.train_on_batch(latent_vect, numpy.ones((batch_size, 1)))
    # GlossGen = G.train_on_batch(noise, numpy.ones((batch_size, 1)))

    Eloss = AE.train_on_batch(warped_A, target_A)

    # print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

    print("%d [D loss: %f]" % (epoch, DlossTrue[0]))

    # *************

    # if epoch % 100 == 0:
    #     save_model_weights()

    figure_A = numpy.stack([
        warped_A,
        target_A,
        encImg,
        fakeImg

    ], axis=1)

    figure = numpy.concatenate([figure_A], axis=0)
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

# ********************************************************************
