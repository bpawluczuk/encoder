import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import binary_crossentropy

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

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)


class VAE:
    _image_shape = (128, 128, 3)
    _latent_dim = 1024
    _batch_size = 16
    _variational = 1

    def _conv(self, filters):
        def block(x):
            x = Conv2D(filters, kernel_size=5, strides=2, padding='same')(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(0.1)(x)
            x = Dropout(0.4)(x)
            return x

        return block

    def _upscale(self, filters):
        def block(x):
            x = Conv2D(filters * 4, kernel_size=3, padding='same')(x)
            x = LeakyReLU(0.1)(x)
            x = PixelShuffler()(x)
            return x

        return block

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._latent_dim), mean=0., stddev=1.)

        return z_mean + K.exp(z_log_var) * epsilon

    def _encoder(self, _input):

        x = self._conv(128)(_input)
        x = self._conv(256)(x)
        x = self._conv(512)(x)
        x = self._conv(1024)(x)
        x = Flatten()(x)

        if not self._variational:
            latent_space = Dense(self._latent_dim)(x)
        else:
            self.z_mean = Dense(self._latent_dim)(x)
            self.z_log_var = Dense(self._latent_dim)(x)

            latent_space = Lambda(self._sampling)([self.z_mean, self.z_log_var])

        x = Dense(8 * 8 * 1024, activation="relu")(latent_space)
        x = Reshape((8, 8, 1024))(x)
        encoder_output = self._upscale(512)(x)
        return Model(_input, encoder_output), self.z_mean, self.z_log_var

    def _decoder(self):

        decoder_input = Input(shape=(16, 16, 512), name="decoder_input")

        x = self._upscale(512)(decoder_input)
        x = self._upscale(256)(x)
        x = self._upscale(128)(x)

        decoder_output = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)
        return Model(decoder_input, decoder_output)

    def _vae_loss(self, input_img, z_decoded):
        input_img = K.flatten(input_img)
        z_decoded = K.flatten(z_decoded)
        xent_loss = binary_crossentropy(input_img, z_decoded)
        kl_loss = - 0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=-1)
        return K.mean(xent_loss + kl_loss)

    def _get_vae(self):

        encoder_input = Input(shape=self._image_shape, name="encoder_input")
        encoder = self._encoder(encoder_input)
        decoder_a = self._decoder()
        decoder_b = self._decoder()

        autoencoder_a = Model(encoder_input, decoder_a(encoder(encoder_input)))
        if not self._variational:
            autoencoder_a.compile(optimizer=optimizer, loss='mean_absolute_error')
        else:
            # loss = self._vae_loss(encoder_input, decoder_a(encoder(encoder_input)))
            # autoencoder_a.add_loss(loss)
            autoencoder_a.compile(optimizer=optimizer, loss=self._vae_loss)

        autoencoder_b = Model(encoder_input, decoder_b(encoder(encoder_input)))
        if not self._variational:
            autoencoder_b.compile(optimizer=optimizer, loss='mean_absolute_error')
        else:
            # loss = self._vae_loss(encoder_input, decoder_b(encoder(encoder_input)))
            # autoencoder_b.add_loss(loss)
            autoencoder_b.compile(optimizer=optimizer, loss=self._vae_loss)

        return encoder, autoencoder_a, autoencoder_a


# ********************************************************************

# encoder, autoencoder_A, autoencoder_B = VAE()._get_vae()

encoder_input = Input(shape=IMAGE_SHAPE, name="encoder_input")
encoder, z_mean, z_log_var = VAE()._encoder(encoder_input)
decoder_a = VAE()._decoder()
decoder_b = VAE()._decoder()


def _vae_loss(input_img, z_decoded):
    input_img = K.flatten(input_img)
    z_decoded = K.flatten(z_decoded)
    xent_loss = binary_crossentropy(input_img, z_decoded)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)


autoencoder_A = Model(encoder_input, decoder_a(encoder(encoder_input)))
autoencoder_A.compile(optimizer=optimizer, loss=_vae_loss)

autoencoder_B = Model(encoder_input, decoder_b(encoder(encoder_input)))
autoencoder_B.compile(optimizer=optimizer, loss=_vae_loss)

# ********************************************************************

# encoder.summary()
autoencoder_A.summary()
# autoencoder_B.summary()

# ********************************************************************

images_A = get_image_paths("dataset/frames/harrison_face")
images_B = get_image_paths("dataset/frames/ryan_face")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

images_A += images_B.mean(axis=(0, 1, 2)) - images_A.mean(axis=(0, 1, 2))

for epoch in range(100000):
    batch_size = 16
    warped_A, target_A = get_training_data(images_A, batch_size)
    warped_B, target_B = get_training_data(images_B, batch_size)

    loss_A = autoencoder_A.train_on_batch(warped_A, target_A)
    loss_B = autoencoder_B.train_on_batch(warped_B, target_B)
    print(epoch, loss_A, loss_B)

    if epoch % 100 == 0:
        # save_model_weights()
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
