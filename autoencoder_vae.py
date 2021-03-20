import cv2
import numpy

from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam

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

optimizer = Adam(lr=5e-5, beta_1=0.5, beta_2=0.999)

class VAE:
    _image_shape = (128, 128, 3)
    _latent_dim = 1024

    def _sampling(self, args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self._latent_dim), mean=0., stddev=1.)

        return z_mean + K.exp(z_log_var) * epsilon

    def _get_z(self):
        input_img = Input(shape=self._image_shape)
        print(K.int_shape(input_img))

        x = Conv2D(128, 5, padding='same', strides=2)(input_img)
        x = LeakyReLU(0.1)(x)
        print(K.int_shape(x))

        x = Conv2D(256, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print(K.int_shape(x))

        x = Conv2D(512, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print(K.int_shape(x))

        x = Conv2D(1024, 5, padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.1)(x)
        x = Dropout(0.4)(x)

        self._shape_before_flattening = K.int_shape(x)

        x = Dense(ENCODER_DIM, activation='relu')(Flatten()(x))
        x = Dense(8 * 8 * 1024, activation='relu')(x)

        self.z_mean = Dense(self._latent_dim)(x)
        self.z_log_var = Dense(self._latent_dim)(x)

        z = Lambda(self._sampling)([self.z_mean, self.z_log_var])

        return input_img, z

    def _get_decoded_z(self, z):

        decoder_input = Input(shape=K.int_shape(z)[1:])
        x = decoder_input
        print("decoder_input" + str(K.int_shape(x)))

        x = Dense(numpy.prod(self._shape_before_flattening[1:]), activation='relu')(decoder_input)
        x = Reshape(self._shape_before_flattening[1:])(x)

        x = Conv2DTranspose(512, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print("decoder next" + str(K.int_shape(x)))

        x = Conv2DTranspose(256, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print("decoder next" + str(K.int_shape(x)))

        x = Conv2DTranspose(128, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print("decoder next" + str(K.int_shape(x)))

        x = Conv2DTranspose(128, 5, padding='same', strides=2)(x)
        x = LeakyReLU(0.1)(x)
        print("decoder next" + str(K.int_shape(x)))

        x = Conv2D(3, kernel_size=5, padding='same', activation='sigmoid')(x)

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
        z_decoded = self._get_decoded_z(z)

        autoencoder_A = Model(input_img, z_decoded)
        loss = self._vae_loss(input_img, z_decoded)
        autoencoder_A.add_loss(loss)
        autoencoder_A.compile(optimizer=optimizer, loss=self._vae_loss, experimental_run_tf_function=False)

        autoencoder_B = Model(input_img, z_decoded)
        loss = self._vae_loss(input_img, z_decoded)
        autoencoder_B.add_loss(loss)
        autoencoder_B.compile(optimizer=optimizer, loss=self._vae_loss, experimental_run_tf_function=False)

        return autoencoder_A, autoencoder_B


# ********************************************************************

images_A = get_image_paths("dataset/frames/harrison_face")
images_B = get_image_paths("dataset/frames/ryan_face")
images_A = load_images(images_A) / 255.0
images_B = load_images(images_B) / 255.0

autoencoder_A, autoencoder_B = VAE()._get_vae()

# ********************************************************************

# encoder.summary()
autoencoder_A.summary()
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
