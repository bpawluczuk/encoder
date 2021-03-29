import cv2
import numpy
from pathlib import Path

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model

from utils import get_image_paths
from pixel_shuffler import PixelShuffler
from util_face import getFaceAndCoordinates
from seamless_image import seamless_images

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


# ******************************************************************************

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

IMAGE_SHAPE = (128, 128, 3)
ENCODER_DIM = 1024


_image_shape = (128, 128, 3)
_latent_dim = 256
_batch_size = 16
_variational = 1
width = 128
height = 128


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


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], _latent_dim), mean=0., stddev=1.)

    return z_mean + K.exp(z_log_var) * epsilon


def vae_loss(input, x_decoded_mean):
    mse_loss = K.mean(keras.losses.mse(input, x_decoded_mean), axis=(1, 2)) * height * width
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    return mse_loss + kl_loss


def Encoder(input_):

    x = conv(128)(input_)
    x = conv(256)(x)
    x = conv(512)(x)
    x = convDropout(512)(x)
    x = Flatten()(x)

    z_mean = Dense(_latent_dim)(x)
    z_log_sigma = Dense(_latent_dim)(x)

    if not _variational:
        latent_space = Dense(_latent_dim)(x)
    else:
        latent_space = Lambda(sampling)([z_mean, z_log_sigma])

    x = Dense(8 * 8 * 512, activation="relu")(latent_space)
    x = Reshape((8, 8, 512))(x)
    x = upscale(512)(x)

    return Model(input_, x), z_log_sigma, z_mean


def Decoder():
    input_ = Input(shape=(16, 16, 512))

    x = upscale(512)(input_)
    x = upscale(256)(x)
    x = upscale(128)(x)

    x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
    return Model(input_, x)


# ********************************************************************

x = Input(shape=IMAGE_SHAPE)

encoder, z_log_sigma, z_mean = Encoder(x)
decoder_A = Decoder()
decoder_B = Decoder()

autoencoder_A = Model(x, decoder_A(encoder(x)))
if not _variational:
    autoencoder_A.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_A.compile(optimizer=optimizer, loss=vae_loss)

autoencoder_B = Model(x, decoder_B(encoder(x)))
if not _variational:
    autoencoder_B.compile(optimizer=optimizer, loss='mean_absolute_error')
else:
    autoencoder_B.compile(optimizer=optimizer, loss=vae_loss)

# ********************************************************************

try:
    if not _variational:
        encoder.load_weights("models/AE/encoder.h5")
        decoder_A.load_weights("models/AE/decoder_a.h5")
        decoder_B.load_weights("models/AE/decoder_b.h5")
    else:
        encoder.load_weights("models/VAE/encoder.h5")
        decoder_A.load_weights("models/VAE/decoder_a.h5")
        decoder_B.load_weights("models/VAE/decoder_b.h5")
    print("... load models")
except:
    print("models does not exist")

images_A = get_image_paths("data/oliwka_512/")
images_B = get_image_paths("data/laura_512/")

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

        cv2.imshow("source", source_image)
        cv2.imshow("source_face", source_image_face)
        cv2.imshow("predict_face", predict_face)
        cv2.imshow("destination_image", destination_image)

        output_file = "output/seamless_new_image.jpg"
        # cv2.imwrite(str(output_file), seamless_destination_image)
        cv2.imshow("seamless_destination_image", seamless_destination_image)

        result = seamless_destination_image

    return result


output_dir = Path('output/oliwka_laura')
output_dir = Path('output/laura_oliwka')
output_dir.mkdir(parents=True, exist_ok=True)


for fn in images_B:

    image = cv2.imread(fn)
    new_image = convert_one_image(autoencoder_A, image)

    if new_image is not None:
        output_file = output_dir / Path(fn).name
        cv2.imwrite(str(output_file), new_image)

key = cv2.waitKey(0)
