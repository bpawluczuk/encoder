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

    # x = Dense(128 * 128 * 3)(input_)
    # x = LeakyReLU(0.1)(x)
    # x = Reshape((128, 128, 3))(x)

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

    # x = Dense(128 * 128 * 3)(input_)
    # x = LeakyReLU(0.1)(x)
    # x = Reshape((128, 128, 3))(x)

    x = Conv2D(128, 5, padding='same')(input_)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(128, 5, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(128, 5, padding='same')(x)
    x = LeakyReLU(0.1)(x)

    x = Conv2D(3, 7, activation='tanh', padding='same')(x)

    return Model(input_, x)


# ********************************************************************


def Discriminator(input_):
    x = Conv2D(128, 3)(input_)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = LeakyReLU()(x)

    x = Conv2D(128, 4, strides=2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Flatten()(x)

    x = Dropout(0.4)(x)
    x = Dense(1, activation="sigmoid")(x)

    return Model(input_, x)


# ********************************************************************

gan_input = keras.Input(shape=IMAGE_SHAPE)

# encoder = Encoder(gan_input)
# decoder = Decoder()
#
# generator = Model(gan_input, decoder(encoder(gan_input)))
# generator.summary()


generator = Generator(gan_input)
generator.summary()


discriminator_input = Input(shape=IMAGE_SHAPE)
discriminator = Discriminator(discriminator_input)
# discriminator.summary()

discriminator_optimizer = keras.optimizers.RMSprop(lr=0.0008, clipvalue=1.0, decay=1e-8)
discriminator.compile(optimizer=discriminator_optimizer, loss='binary_crossentropy')

discriminator.trainable = False

gan_input = keras.Input(shape=IMAGE_SHAPE)
gan_output = discriminator(generator(gan_input))
gan = keras.models.Model(gan_input, gan_output)

gan_optimizer = keras.optimizers.RMSprop(lr=0.0004, clipvalue=1.0, decay=1e-8)
gan.compile(optimizer=gan_optimizer, loss='binary_crossentropy')

# gan.summary()

# ********************************************************************

batch_size = 16

valid = numpy.ones((batch_size, 1))
fake = numpy.zeros((batch_size, 1))

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
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

for epoch in range(10000000):
    warped_A, target_A = get_training_data(images_A, batch_size)

    # random_latent_vectors = numpy.random.normal(size=(batch_size, IMAGE_SHAPE))
    generated_images = generator.predict(warped_A)

    combined_images = numpy.concatenate([generated_images, target_A])

    labels = numpy.concatenate([valid, fake])
    labels += 0.05 * numpy.random.random(labels.shape)

    d_loss = discriminator.train_on_batch(combined_images, labels)

    # random_latent_vectors = numpy.random.normal(size=(batch_size, 128))
    misleading_targets = numpy.zeros((batch_size, 1))

    a_loss = gan.train_on_batch(warped_A, misleading_targets)

    print('strata dyskryminatora w kroku %s: %s' % (epoch, d_loss))
    print('strata przeciwna: %s: %s' % (epoch, a_loss))

    # *************

    if epoch % 100 == 0:
        save_model_weights()

    figure_A = numpy.stack([
        warped_A,
        target_A,
        generated_images,
    ], axis=1)

    figure = numpy.concatenate([figure_A], axis=0)
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)

# ********************************************************************
