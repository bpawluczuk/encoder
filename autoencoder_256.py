import numpy
import cv2

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, Reshape, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

from pixel_shuffler import PixelShuffler
from training_data import get_training_data
from utils import get_image_paths, load_images, stack_images

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# ********************************************************************

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ********************************************************************

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)


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


def Encoder():
    input_ = Input(shape=IMAGE_SHAPE)
    x = input_
    x = conv(128)(x)
    x = conv(256)(x)
    x = conv(512)(x)
    x = conv(1024)(x)
    x = Dense(ENCODER_DIM)(Flatten()(x))
    x = Dense(8 * 8 * 1024)(x)
    x = Reshape((8, 8, 1024))(x)
    x = upscale(512)(x)
    return Model(input_, x)


def Decoder():
    input_ = Input(shape=(16, 16, 512))
    x = input_
    x = upscale(512)(x)
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


# ********************************************************************

def save_model_weights():
    encoder.save_weights("models/128/encoder.h5")
    decoder_A.save_weights("models/128/decoder_A.h5")
    decoder_B.save_weights("models/128/decoder_B.h5")
    print("save model weights")


# ********************************************************************

encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

autoencoder_A.summary()

try:
    encoder.load_weights("models/128/encoder.h5")
    decoder_A.load_weights("models/128/decoder_A.h5")
    decoder_B.load_weights("models/128/decoder_B.h5")
    print("... load models")
except:
    print("models does not exist")


images_A = get_image_paths("dataset/frames/harrison_face")
images_B = get_image_paths("dataset/frames/ryan_face")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))


for epoch in range(100000):
    batch_size = 32
    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
        save_model_weights()
        test_A = target_A[0:14]
        test_B = target_B[0:14]

    figure_A = numpy.stack([
        test_A,
        autoencoder_A.predict(test_A),
        autoencoder_B.predict(test_A),
    ], axis=1)

    figure_B = numpy.stack([
        test_B,
        autoencoder_B.predict(test_B),
        autoencoder_A.predict(test_B),
    ], axis=1)

    figure = numpy.concatenate([figure_A, figure_B], axis=0)
    figure = figure.reshape((4, 7) + figure.shape[1:])
    figure = stack_images(figure)

    figure = numpy.clip(figure * 255, 0, 255).astype('uint8')

    cv2.imshow("", figure)
    key = cv2.waitKey(1)
