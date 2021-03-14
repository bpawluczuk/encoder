import os
import cv2
import numpy
from pathlib import Path

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, Dense, Flatten, Reshape, BatchNormalization, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

from pixel_shuffler import PixelShuffler
from seamless_image import seamless_images
from utils import get_image_paths
from util_face import getFaceAndCoordinates

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# ******************************************************************************

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024


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
    encoder.load_weights("models/128/encoder.h5")
    decoder_A.load_weights("models/128/decoder_A.h5")
    decoder_B.load_weights("models/128/decoder_B.h5")

    print("... load models")
except:
    print("models does not exist")

source_dir = "dataset/frames/harrison_512/"
dest_dir = "dataset/frames/harrison_face/"

images_A = get_image_paths("data/harrison/")
images_B = get_image_paths("data/ryan/")
# images_A = get_image_paths("dataset/frames/harrison_512/")
# images_B = get_image_paths("dataset/frames/ryan_512/")


def convert_one_image(autoencoder, source_image):
    assert source_image.shape == (512, 512, 3)

    resultFace = getFaceAndCoordinates(source_image)

    result = None

    if resultFace is not None:
        xmin, ymin, xmax, ymax, h, w, face = resultFace

        source_image_face = cv2.resize(face, (int(128), int(128)))
        source_image_face_expand_dims = numpy.expand_dims(source_image_face, 0)
        predict_face = autoencoder.predict(source_image_face_expand_dims / 255.0)[0]
        predict_face = numpy.clip(predict_face * 255, 0, 255).astype(image.dtype)
        predict_face = cv2.resize(predict_face, (xmax - xmin, ymax - ymin))
        destination_image = source_image.copy()
        destination_image[ymin:ymin + h, xmin:xmin + w] = predict_face
        seamless_destination_image = seamless_images(destination_image, source_image)

        # cv2.imshow("source_image_face", source_image_face
        # cv2.imshow("predict_face", predict_face)
        # cv2.imshow("destination_image", destination_image)

        output_file = "output/seamless_new_image.jpg"
        # cv2.imwrite(str(output_file), seamless_destination_image)
        # cv2.imshow("seamless_destination_image", seamless_destination_image)

        result = seamless_destination_image

    return result


output_dir = Path('output')
output_dir.mkdir(parents=True, exist_ok=True)


# for fn in images_A:

    # image = cv2.imread(fn)
    # xmin, ymin, xmax, ymax, h, w, face = getFaceAndCoordinates(image)
    #
    # if face is not None:
    #     cv2.imwrite(dest_dir + Path(fn).name, cv2.resize(face, (int(128), int(128))))
    # else:
    #     if os.path.exists(source_dir + Path(fn).name):
    #         os.remove(source_dir + Path(fn).name)
    #         print(source_dir + Path(fn).name)
    #     else:
    #         print("The file does not exist")

for fn in images_A:

    image = cv2.imread(fn)
    new_image = convert_one_image(autoencoder_B, image)

    if new_image is not None:
        output_file = output_dir / Path(fn).name
        cv2.imwrite(str(output_file), new_image)

key = cv2.waitKey(0)
