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
import dlib

frontal_face_detector = dlib.get_frontal_face_detector()
frontal_face_predictor = dlib.shape_predictor("detect/shape_predictor_68_face_landmarks.dat")

def getFace(source_image):
    source_image_grayscale = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY)
    source_image_canvas = numpy.zeros_like(source_image_grayscale)
    source_faces = frontal_face_detector(source_image_grayscale)

    for source_face in source_faces:

        source_face_landmarks = frontal_face_predictor(source_image_grayscale, source_face)
        source_face_landmark_points = []

        for landmark_no in range(0, 68):
            x_point = source_face_landmarks.part(landmark_no).x
            y_point = source_face_landmarks.part(landmark_no).y
            source_face_landmark_points.append((x_point, y_point))

        source_face_landmark_points_array = numpy.array(source_face_landmark_points, numpy.int32)

        source_face_convexhull = cv2.convexHull(source_face_landmark_points_array)
        cv2.fillConvexPoly(source_image_canvas, source_face_convexhull, 255)
        # cv2.imwrite(dest_dir_canvas + name + "_face_canvas_" + str(inc) + ".jpg", source_image_canvas)

        # place the created mask over the source image
        source_face_image = cv2.bitwise_and(source_image, source_image, mask=source_image_canvas)
        # cv2.imwrite(dest_dir_face + name + "_face_" + str(inc) + ".jpg", source_face_image)
        return source_face_image


# ******************************************************************************

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

IMAGE_SHAPE = (64, 64, 3)
ENCODER_DIM = 1024


def conv(filters):
    def block(x):
        x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
        # x = BatchNormalization()(x)
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

encoder.load_weights("models/encoder.h5")
decoder_A.load_weights("models/decoder_A.h5")
decoder_B.load_weights("models/decoder_B.h5")

images_A = get_image_paths("data/bruce")
images_B = get_image_paths("data/matt")


def convert_one_image(autoencoder, image):
    assert image.shape == (256, 256, 3)
    crop = slice(48, 208)
    face = image[48:208, 48:208]
    print(face.shape)
    face = cv2.resize(face, (64, 64))
    print(face.shape)
    face = numpy.expand_dims(face, 0)
    print(face.shape)

    # cv2.imshow("", getFace(image))

    new_face = autoencoder.predict(face / 255.0)[0]

    new_face = numpy.clip(new_face * 255, 0, 255).astype(image.dtype)
    new_face = cv2.resize(new_face, (160, 160))
    new_image = image.copy()
    new_image[crop, crop] = new_face

    q = getFace(new_face)
    cv2.imshow("", getFace(new_face))
    return new_image



output_dir = Path('output')
output_dir.mkdir(parents=True, exist_ok=True)

for fn in images_B:
    image = cv2.imread(fn)
    new_image = convert_one_image(autoencoder_A, image)
    output_file = output_dir / Path(fn).name
    cv2.imwrite(str(output_file), new_image)
    # cv2.imshow("", new_image)

key = cv2.waitKey(0)
