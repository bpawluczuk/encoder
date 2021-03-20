import cv2
import numpy

from tensorflow.keras.layers import *

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import binary_crossentropy

from utils import get_image_paths, load_images, stack_images
from training_data import get_training_data

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


class VAE:
    _image_shape = (128, 128, 3)
    _latent_dim = 4

    def _get_z(self):
        input_img = Input(shape=self._image_shape)

        x = Conv2D(32, 3, padding='same')(input_img)
        x = LeakyReLU(0.1)(x)

        print(K.int_shape(x))

        x = Conv2D(64, 3, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = LeakyReLU(0.1)(x)

        x = Conv2D(64, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.4)(x)

        self._shape_before_flattening = K.int_shape(x)
        print(K.int_shape(x))

        x = Flatten()(x)
        print(K.int_shape(x))
        x = Dense(64, activation='relu')(x)

        self.z_mean = Dense(self._latent_dim)(x)
        self.z_log_var = Dense(self._latent_dim)(x)

        z = Lambda(self._sampling)([self.z_mean, self.z_log_var])

        return input_img, z

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._latent_dim), mean=0., stddev=1.)

        return z_mean + K.exp(z_log_var) * epsilon

    def _get_decoded_z(self, z):
        """ Decoder Network that decodes a latent space point z into an image """
        decoder_input = Input(K.int_shape(z)[1:])

        # Use a dense layer to psample to the correct number of units (exclude batch size with [1:])
        x = Dense(numpy.prod(self._shape_before_flattening[1:]), activation='relu')(decoder_input)

        # Reshape into an image of the same shape as before the last Flatten layer in the encoder
        x = Reshape(self._shape_before_flattening[1:])(x)

        # Apply reverse operation to the initial stack of Conv2D: a Conv2DTranspose
        x = Conv2DTranspose(32, 3, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)

        # We end up with a feature map of the same size as the original input
        x = Conv2D(self._image_shape[-1], 3, padding='same', activation='sigmoid')(x)

        decoder = Model(decoder_input, x)
        z_decoded = decoder(z)

        return z_decoded

    def _vae_loss(self, input_img, z_decoded):
        input_img = K.flatten(input_img)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(input_img, z_decoded)
        kl_loss = -5e-4 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def _get_vae(self):

        input_img, z = self._get_z()

        z_decoded_A = self._get_decoded_z(z)
        vae_A = Model(input_img, z_decoded_A)
        loss = self._vae_loss(input_img, z_decoded_A)
        vae_A.add_loss(loss)
        vae_A.compile(optimizer='rmsprop', loss=self._vae_loss, experimental_run_tf_function=False)

        z_decoded_B = self._get_decoded_z(z)
        vae_B = Model(input_img, z_decoded_B)
        loss = self._vae_loss(input_img, z_decoded_B)
        vae_B.add_loss(loss)
        vae_B.compile(optimizer='rmsprop', loss=self._vae_loss, experimental_run_tf_function=False)

        return vae_A, vae_A

# ********************************************************************

images_A = get_image_paths("dataset/frames/harrison_face")
images_B = get_image_paths("dataset/frames/ryan_face")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

autoencoder_A, autoencoder_B = VAE()._get_vae()

# ********************************************************************

# encoder.summary()
# autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

for epoch in range(10000):
    batch_size = 16
    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
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

# ********************************************************************