import cv2
import numpy
from pathlib import Path

from utils import get_image_paths

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from pixel_shuffler import PixelShuffler

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ******************************************************************************

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024


def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        x = LeakyReLU(0.1)(x)
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
    x = Dense(4 * 4 * 1024)(x)
    x = Reshape((4, 4, 1024))(x)
    x = upscale(512)(x)
    return Model(input_, x)


def Decoder():
    input_ = Input(shape=(8, 8, 512))
    x = input_
    x = upscale(256)(x)
    x = upscale(128)(x)
    x = upscale(64)(x)
    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


encoder = Encoder()
decoder_A = Decoder()
decoder_B = Decoder()

x = Input(shape=IMAGE_SHAPE)

autoencoder_A = Model(x, decoder_A(encoder(x)))
autoencoder_B = Model(x, decoder_B(encoder(x)))
autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')

# **********************************************************

try:
    encoder.load_weights("models/64/encoder.h5")
    decoder_A.load_weights("models/64/decoder_A.h5")
    decoder_B.load_weights("models/64/decoder_B.h5")
except:
    print("models does not exist")

images_A = get_image_paths("data/bruce")
images_B = get_image_paths("data/matt")


def convert_one_image(autoencoder, image):
    assert image.shape == (256, 256, 3)
    crop = slice(48, 208)
    face = image[48:208, 48:208]

    face = cv2.resize(face, (64, 64))
    face = numpy.expand_dims(face, 0)

    new_face = autoencoder.predict(face / 255.0)[0]
    new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)
    new_face = cv2.resize(new_face, (160, 160))
    new_image = image.copy()

    cv2.imshow("four", new_face)

    new_image[crop, crop] = new_face

    cv2.imshow("five", new_image)


output_dir = Path('output')
output_dir.mkdir(parents=True, exist_ok=True)

for fn in images_B:
    image = cv2.imread(fn)
    new_image = convert_one_image(autoencoder_A, image)
    output_file = output_dir / Path(fn).name
    cv2.imwrite(str(output_file), new_image)
    # cv2.imshow("", new_image)

key = cv2.waitKey(0)
